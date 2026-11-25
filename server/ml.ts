import type {
  ContributingFactor,
  FeatureImportanceItem,
  ModelEvaluation,
  ModelKey,
  PredictionInput,
  PredictionResult,
  PreprocessingConfig,
  RawDatasetRow,
  RiskCategory,
  RocPoint,
  TrainedModelsBundle,
} from '../src/types.ts';
import { createSeededRng } from './dataset.ts';
import {
  deduplicateRows,
  ENCODING_MAPS,
  fitPreprocessorOnRows,
  MODEL_FEATURE_KEYS,
  MODEL_FEATURE_LABELS,
  RISK_CLASSES,
  stratifiedSplit,
  transformRowToScaledVector,
  type FittedPreprocessor,
} from './preprocessing.ts';

const NUM_CLASSES = 3;

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(Math.max(-50, Math.min(50, v - maxLogit))));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

function normalizeImportance(raw: number[]): FeatureImportanceItem[] {
  const sum = raw.reduce((a, b) => a + Math.abs(b), 0) || 1;
  return MODEL_FEATURE_KEYS.map((feature, idx) => ({
    feature,
    label: MODEL_FEATURE_LABELS[feature] ?? feature,
    importance: Math.round((Math.abs(raw[idx]) / sum) * 10000) / 10000,
  })).sort((a, b) => b.importance - a.importance);
}

// ============================================================================
// 1. Multinomial Logistic Regression (Softmax + L2 Regularization)
// ============================================================================
interface LogisticModel {
  weights: number[][]; // [NUM_CLASSES][nFeatures]
  biases: number[]; // [NUM_CLASSES]
  featureImportance: number[];
}

function trainLogisticRegression(
  X: number[][],
  y: number[],
  epochs = 450,
  lr = 0.25,
  l2 = 0.02
): LogisticModel {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  const weights: number[][] = Array.from({ length: NUM_CLASSES }, () =>
    new Array(nFeatures).fill(0)
  );
  const biases = new Array(NUM_CLASSES).fill(0);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const gradW: number[][] = Array.from({ length: NUM_CLASSES }, () =>
      new Array(nFeatures).fill(0)
    );
    const gradB = new Array(NUM_CLASSES).fill(0);

    for (let i = 0; i < nSamples; i++) {
      const x = X[i];
      const logits = new Array(NUM_CLASSES).fill(0);
      for (let c = 0; c < NUM_CLASSES; c++) {
        let z = biases[c];
        for (let f = 0; f < nFeatures; f++) {
          z += weights[c][f] * x[f];
        }
        logits[c] = z;
      }
      const probs = softmax(logits);
      for (let c = 0; c < NUM_CLASSES; c++) {
        const diff = probs[c] - (y[i] === c ? 1 : 0);
        gradB[c] += diff;
        for (let f = 0; f < nFeatures; f++) {
          gradW[c][f] += diff * x[f];
        }
      }
    }

    const step = lr / (1 + 0.002 * epoch);
    for (let c = 0; c < NUM_CLASSES; c++) {
      biases[c] -= (step * gradB[c]) / nSamples;
      for (let f = 0; f < nFeatures; f++) {
        weights[c][f] -= step * (gradW[c][f] / nSamples + l2 * weights[c][f]);
      }
    }
  }

  const featureImportance = new Array(nFeatures).fill(0);
  for (let f = 0; f < nFeatures; f++) {
    let normSq = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      normSq += weights[c][f] * weights[c][f];
    }
    featureImportance[f] = Math.sqrt(normSq);
  }

  return { weights, biases, featureImportance };
}

function predictProbaLogistic(model: LogisticModel, x: number[]): number[] {
  const logits = new Array(NUM_CLASSES).fill(0);
  for (let c = 0; c < NUM_CLASSES; c++) {
    let z = model.biases[c];
    for (let f = 0; f < x.length; f++) {
      z += model.weights[c][f] * x[f];
    }
    logits[c] = z;
  }
  return softmax(logits);
}

// ============================================================================
// 2. Multi-Class CART Decision Tree Classifier (Gini Impurity)
// ============================================================================
interface CartNode {
  isLeaf: boolean;
  probs: number[];
  featureIdx?: number;
  threshold?: number;
  left?: CartNode;
  right?: CartNode;
}

interface DecisionTreeModel {
  root: CartNode;
  featureImportance: number[];
}

function giniFromCounts(counts: number[], total: number): number {
  if (total === 0) return 0;
  let sumSq = 0;
  for (let c = 0; c < counts.length; c++) {
    const p = counts[c] / total;
    sumSq += p * p;
  }
  return 1 - sumSq;
}

function buildCartTree(
  X: number[][],
  y: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  featureImportance: number[],
  maxFeatures: number | null,
  rng: () => number
): CartNode {
  const counts = new Array(NUM_CLASSES).fill(0);
  for (const idx of indices) counts[y[idx]]++;
  const n = indices.length;
  const probs = counts.map((c) => (n > 0 ? c / n : 1 / NUM_CLASSES));

  const parentGini = giniFromCounts(counts, n);
  if (depth >= maxDepth || n < minSamplesSplit || parentGini < 1e-7) {
    return { isLeaf: true, probs };
  }

  const nFeatures = X[0].length;
  let candidateFeatures = Array.from({ length: nFeatures }, (_, i) => i);
  if (maxFeatures !== null && maxFeatures < nFeatures) {
    for (let i = candidateFeatures.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = candidateFeatures[i];
      candidateFeatures[i] = candidateFeatures[j];
      candidateFeatures[j] = tmp;
    }
    candidateFeatures = candidateFeatures.slice(0, maxFeatures);
  }

  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (const f of candidateFeatures) {
    const pairs = indices.map((idx) => ({ val: X[idx][f], cls: y[idx], idx }));
    pairs.sort((a, b) => a.val - b.val);

    const leftCounts = new Array(NUM_CLASSES).fill(0);
    const rightCounts = [...counts];

    for (let i = 0; i < n - 1; i++) {
      const cls = pairs[i].cls;
      leftCounts[cls]++;
      rightCounts[cls]--;

      const leftN = i + 1;
      const rightN = n - leftN;
      if (leftN < minSamplesLeaf || rightN < minSamplesLeaf) continue;
      if (pairs[i].val === pairs[i + 1].val) continue;

      const leftGini = giniFromCounts(leftCounts, leftN);
      const rightGini = giniFromCounts(rightCounts, rightN);
      const weightedGini = (leftN / n) * leftGini + (rightN / n) * rightGini;
      const gain = parentGini - weightedGini;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = (pairs[i].val + pairs[i + 1].val) / 2;
      }
    }
  }

  if (bestFeature === -1 || bestGain <= 1e-7) {
    return { isLeaf: true, probs };
  }

  for (const idx of indices) {
    if (X[idx][bestFeature] <= bestThreshold) bestLeft.push(idx);
    else bestRight.push(idx);
  }

  if (bestLeft.length < minSamplesLeaf || bestRight.length < minSamplesLeaf) {
    return { isLeaf: true, probs };
  }

  featureImportance[bestFeature] += n * bestGain;

  return {
    isLeaf: false,
    probs,
    featureIdx: bestFeature,
    threshold: bestThreshold,
    left: buildCartTree(
      X,
      y,
      bestLeft,
      depth + 1,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      featureImportance,
      maxFeatures,
      rng
    ),
    right: buildCartTree(
      X,
      y,
      bestRight,
      depth + 1,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      featureImportance,
      maxFeatures,
      rng
    ),
  };
}

function trainDecisionTree(
  X: number[][],
  y: number[],
  maxDepth = 6,
  minSamplesSplit = 6,
  minSamplesLeaf = 2,
  seed = 42
): DecisionTreeModel {
  const nFeatures = X[0].length;
  const featureImportance = new Array(nFeatures).fill(0);
  const indices = Array.from({ length: X.length }, (_, i) => i);
  const rng = createSeededRng(seed);
  const root = buildCartTree(
    X,
    y,
    indices,
    0,
    maxDepth,
    minSamplesSplit,
    minSamplesLeaf,
    featureImportance,
    null,
    rng
  );
  return { root, featureImportance };
}

function predictProbaTree(node: CartNode, x: number[]): number[] {
  let curr = node;
  while (!curr.isLeaf && curr.featureIdx !== undefined && curr.threshold !== undefined) {
    curr = x[curr.featureIdx] <= curr.threshold ? curr.left! : curr.right!;
  }
  return curr.probs;
}

// ============================================================================
// 3. Multi-Class Random Forest Classifier (Bagging + Feature Subsampling)
// ============================================================================
interface RandomForestModel {
  trees: CartNode[];
  featureImportance: number[];
}

function trainRandomForest(
  X: number[][],
  y: number[],
  nEstimators = 35,
  maxDepth = 8,
  seed = 42
): RandomForestModel {
  const n = X.length;
  const nFeatures = X[0].length;
  const maxFeatures = Math.max(2, Math.round(Math.sqrt(nFeatures)));
  const featureImportance = new Array(nFeatures).fill(0);
  const rng = createSeededRng(seed + 101);
  const trees: CartNode[] = [];

  for (let t = 0; t < nEstimators; t++) {
    const bootIndices: number[] = [];
    for (let i = 0; i < n; i++) {
      bootIndices.push(Math.floor(rng() * n));
    }
    const tree = buildCartTree(
      X,
      y,
      bootIndices,
      0,
      maxDepth,
      5,
      2,
      featureImportance,
      maxFeatures,
      rng
    );
    trees.push(tree);
  }

  return { trees, featureImportance };
}

function predictProbaRandomForest(model: RandomForestModel, x: number[]): number[] {
  const avg = new Array(NUM_CLASSES).fill(0);
  for (const tree of model.trees) {
    const p = predictProbaTree(tree, x);
    for (let c = 0; c < NUM_CLASSES; c++) avg[c] += p[c];
  }
  return avg.map((v) => v / model.trees.length);
}

// ============================================================================
// 4. Multi-Class Gradient Boosting Classifier (Stagewise Softmax GBDT)
// ============================================================================
interface RegTreeNode {
  isLeaf: boolean;
  value: number;
  featureIdx?: number;
  threshold?: number;
  left?: RegTreeNode;
  right?: RegTreeNode;
}

interface GradientBoostingModel {
  initLogits: number[];
  learningRate: number;
  stages: RegTreeNode[][]; // [stage][class]
  featureImportance: number[];
}

function buildRegTree(
  X: number[][],
  residuals: number[],
  probs: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamplesLeaf: number,
  featureImportance: number[]
): RegTreeNode {
  const n = indices.length;
  // Newton-Raphson optimal leaf step for multi-class log-loss
  let num = 0;
  let den = 0;
  for (const idx of indices) {
    num += residuals[idx];
    const p = probs[idx];
    den += Math.max(1e-6, p * (1 - p));
  }
  const leafValue = ((NUM_CLASSES - 1) / NUM_CLASSES) * (num / Math.max(1e-5, den));

  if (depth >= maxDepth || n < minSamplesLeaf * 2) {
    return { isLeaf: true, value: Math.max(-3, Math.min(3, leafValue)) };
  }

  const totalSum = indices.reduce((acc, idx) => acc + residuals[idx], 0);
  const parentScore = (totalSum * totalSum) / n;

  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  const nFeatures = X[0].length;

  for (let f = 0; f < nFeatures; f++) {
    const pairs = indices.map((idx) => ({ val: X[idx][f], r: residuals[idx] }));
    pairs.sort((a, b) => a.val - b.val);

    let leftSum = 0;
    for (let i = 0; i < n - 1; i++) {
      leftSum += pairs[i].r;
      const leftN = i + 1;
      const rightN = n - leftN;
      if (leftN < minSamplesLeaf || rightN < minSamplesLeaf) continue;
      if (pairs[i].val === pairs[i + 1].val) continue;

      const rightSum = totalSum - leftSum;
      const gain = (leftSum * leftSum) / leftN + (rightSum * rightSum) / rightN - parentScore;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = (pairs[i].val + pairs[i + 1].val) / 2;
      }
    }
  }

  if (bestFeature === -1 || bestGain <= 1e-7) {
    return { isLeaf: true, value: Math.max(-3, Math.min(3, leafValue)) };
  }

  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (const idx of indices) {
    if (X[idx][bestFeature] <= bestThreshold) leftIdx.push(idx);
    else rightIdx.push(idx);
  }

  if (leftIdx.length < minSamplesLeaf || rightIdx.length < minSamplesLeaf) {
    return { isLeaf: true, value: Math.max(-3, Math.min(3, leafValue)) };
  }

  featureImportance[bestFeature] += bestGain;

  return {
    isLeaf: false,
    value: leafValue,
    featureIdx: bestFeature,
    threshold: bestThreshold,
    left: buildRegTree(
      X,
      residuals,
      probs,
      leftIdx,
      depth + 1,
      maxDepth,
      minSamplesLeaf,
      featureImportance
    ),
    right: buildRegTree(
      X,
      residuals,
      probs,
      rightIdx,
      depth + 1,
      maxDepth,
      minSamplesLeaf,
      featureImportance
    ),
  };
}

function predictRegTree(node: RegTreeNode, x: number[]): number {
  let curr = node;
  while (!curr.isLeaf && curr.featureIdx !== undefined && curr.threshold !== undefined) {
    curr = x[curr.featureIdx] <= curr.threshold ? curr.left! : curr.right!;
  }
  return curr.value;
}

function trainGradientBoosting(
  X: number[][],
  y: number[],
  nEstimators = 38,
  learningRate = 0.12,
  maxDepth = 3
): GradientBoostingModel {
  const n = X.length;
  const nFeatures = X[0].length;
  const featureImportance = new Array(nFeatures).fill(0);

  const classCounts = new Array(NUM_CLASSES).fill(0);
  for (const cls of y) classCounts[cls]++;
  const initLogits = classCounts.map((c) => Math.log(Math.max(1e-5, c / n)));

  const F: number[][] = Array.from({ length: n }, () => [...initLogits]);
  const allIndices = Array.from({ length: n }, (_, i) => i);
  const stages: RegTreeNode[][] = [];

  for (let m = 0; m < nEstimators; m++) {
    const probsMatrix = F.map((row) => softmax(row));
    const stageTrees: RegTreeNode[] = [];

    for (let c = 0; c < NUM_CLASSES; c++) {
      const residuals = new Array(n);
      const classProbs = new Array(n);
      for (let i = 0; i < n; i++) {
        const p = probsMatrix[i][c];
        classProbs[i] = p;
        residuals[i] = (y[i] === c ? 1 : 0) - p;
      }
      const tree = buildRegTree(
        X,
        residuals,
        classProbs,
        allIndices,
        0,
        maxDepth,
        3,
        featureImportance
      );
      stageTrees.push(tree);
      for (let i = 0; i < n; i++) {
        F[i][c] += learningRate * predictRegTree(tree, X[i]);
      }
    }
    stages.push(stageTrees);
  }

  return { initLogits, learningRate, stages, featureImportance };
}

function predictProbaGradientBoosting(
  model: GradientBoostingModel,
  x: number[]
): number[] {
  const logits = [...model.initLogits];
  for (const stage of model.stages) {
    for (let c = 0; c < NUM_CLASSES; c++) {
      logits[c] += model.learningRate * predictRegTree(stage[c], x);
    }
  }
  return softmax(logits);
}

// ============================================================================
// Evaluation Metrics (Confusion Matrix, Precision, Recall, F1, ROC, AUC)
// ============================================================================
function computeBinaryRoc(
  yTrueBinary: number[],
  yScores: number[]
): { curve: RocPoint[]; auc: number } {
  const pairs = yTrueBinary.map((label, i) => ({ label, score: yScores[i] }));
  pairs.sort((a, b) => b.score - a.score);

  const totalPos = pairs.reduce((s, p) => s + p.label, 0);
  const totalNeg = pairs.length - totalPos;

  if (totalPos === 0 || totalNeg === 0) {
    return {
      curve: [
        { fpr: 0, tpr: 0 },
        { fpr: 1, tpr: 1 },
      ],
      auc: 0.5,
    };
  }

  const curve: RocPoint[] = [{ fpr: 0, tpr: 0 }];
  let tp = 0;
  let fp = 0;

  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i].label === 1) tp++;
    else fp++;

    const nextDiff = i === pairs.length - 1 || pairs[i + 1].score !== pairs[i].score;
    if (nextDiff) {
      curve.push({
        fpr: Math.round((fp / totalNeg) * 10000) / 10000,
        tpr: Math.round((tp / totalPos) * 10000) / 10000,
      });
    }
  }

  let auc = 0;
  for (let i = 1; i < curve.length; i++) {
    const dx = curve[i].fpr - curve[i - 1].fpr;
    const avgY = (curve[i].tpr + curve[i - 1].tpr) / 2;
    auc += dx * avgY;
  }

  return { curve, auc: Math.round(auc * 10000) / 10000 };
}

function interpolateTpr(curve: RocPoint[], targetFpr: number): number {
  for (let i = 1; i < curve.length; i++) {
    if (curve[i].fpr >= targetFpr) {
      const x0 = curve[i - 1].fpr;
      const x1 = curve[i].fpr;
      const y0 = curve[i - 1].tpr;
      const y1 = curve[i].tpr;
      if (x1 - x0 < 1e-9) return y1;
      return y0 + ((targetFpr - x0) / (x1 - x0)) * (y1 - y0);
    }
  }
  return 1;
}

function evaluatePredictions(
  key: ModelKey,
  name: string,
  shortName: string,
  description: string,
  hyperparameters: Record<string, string | number>,
  yTest: number[],
  probaTest: number[][],
  rawImportance: number[],
  trainSize: number,
  trainingTimeMs: number
): ModelEvaluation {
  const n = yTest.length;
  const cm: number[][] = Array.from({ length: NUM_CLASSES }, () =>
    new Array(NUM_CLASSES).fill(0)
  );
  let correct = 0;

  for (let i = 0; i < n; i++) {
    const probs = probaTest[i];
    let pred = 0;
    for (let c = 1; c < NUM_CLASSES; c++) {
      if (probs[c] > probs[pred]) pred = c;
    }
    cm[yTest[i]][pred]++;
    if (pred === yTest[i]) correct++;
  }

  const accuracy = n > 0 ? Math.round((correct / n) * 10000) / 10000 : 0;

  const perClassMetrics = {} as ModelEvaluation['perClassMetrics'];
  const perClassRocCurves = {} as ModelEvaluation['perClassRocCurves'];

  let sumP = 0;
  let sumR = 0;
  let sumF1 = 0;
  let wP = 0;
  let wR = 0;
  let wF1 = 0;
  let sumAuc = 0;

  for (let c = 0; c < NUM_CLASSES; c++) {
    const tp = cm[c][c];
    let fp = 0;
    let fn = 0;
    let support = 0;
    for (let r = 0; r < NUM_CLASSES; r++) {
      if (r !== c) fp += cm[r][c];
      support += cm[c][r];
    }
    fn = support - tp;

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    const binaryTrue = yTest.map((label) => (label === c ? 1 : 0));
    const classScores = probaTest.map((p) => p[c]);
    const roc = computeBinaryRoc(binaryTrue, classScores);

    const riskName = RISK_CLASSES[c];
    perClassMetrics[riskName] = {
      precision: Math.round(precision * 10000) / 10000,
      recall: Math.round(recall * 10000) / 10000,
      f1: Math.round(f1 * 10000) / 10000,
      support,
      auc: roc.auc,
    };
    perClassRocCurves[riskName] = roc.curve;

    sumP += precision;
    sumR += recall;
    sumF1 += f1;
    wP += precision * support;
    wR += recall * support;
    wF1 += f1 * support;
    sumAuc += roc.auc;
  }

  // Macro-average ROC curve across 25 uniform FPR grid points
  const macroRocCurve: RocPoint[] = [];
  const gridSteps = 25;
  for (let s = 0; s <= gridSteps; s++) {
    const fpr = Math.round((s / gridSteps) * 10000) / 10000;
    let tprSum = 0;
    for (const cls of RISK_CLASSES) {
      tprSum += interpolateTpr(perClassRocCurves[cls], fpr);
    }
    macroRocCurve.push({
      fpr,
      tpr: Math.round((tprSum / NUM_CLASSES) * 10000) / 10000,
    });
  }

  return {
    key,
    name,
    shortName,
    description,
    hyperparameters,
    accuracy,
    precisionMacro: Math.round((sumP / NUM_CLASSES) * 10000) / 10000,
    recallMacro: Math.round((sumR / NUM_CLASSES) * 10000) / 10000,
    f1Macro: Math.round((sumF1 / NUM_CLASSES) * 10000) / 10000,
    precisionWeighted: n > 0 ? Math.round((wP / n) * 10000) / 10000 : 0,
    recallWeighted: n > 0 ? Math.round((wR / n) * 10000) / 10000 : 0,
    f1Weighted: n > 0 ? Math.round((wF1 / n) * 10000) / 10000 : 0,
    macroAuc: Math.round((sumAuc / NUM_CLASSES) * 10000) / 10000,
    perClassMetrics,
    confusionMatrix: cm,
    classLabels: RISK_CLASSES,
    macroRocCurve,
    perClassRocCurves,
    featureImportance: normalizeImportance(rawImportance),
    trainSize,
    testSize: n,
    trainingTimeMs: Math.round(trainingTimeMs * 10) / 10,
  };
}

// ============================================================================
// Trained Runtime Engine
// ============================================================================
export interface TrainedRuntimeState {
  bundle: TrainedModelsBundle;
  fittedPreprocessor: FittedPreprocessor;
  logistic: LogisticModel;
  decisionTree: DecisionTreeModel;
  randomForest: RandomForestModel;
  gradientBoosting: GradientBoostingModel;
}

export function trainAllModels(
  rawRows: RawDatasetRow[],
  datasetName: string,
  isSynthetic: boolean,
  config: PreprocessingConfig
): TrainedRuntimeState {
  const workingRows = config.removeDuplicates
    ? deduplicateRows(rawRows).uniqueRows
    : rawRows;

  const validRows: RawDatasetRow[] = [];
  const labels: number[] = [];

  for (const r of workingRows) {
    const rawRisk = String(r.risk_category ?? '').trim().toLowerCase();
    let cls: number | null = null;
    if (rawRisk === 'low risk' || rawRisk === 'low' || rawRisk === '0') cls = 0;
    else if (
      rawRisk === 'moderate risk' ||
      rawRisk === 'moderate' ||
      rawRisk === 'medium' ||
      rawRisk === '1'
    ) {
      cls = 1;
    } else if (rawRisk === 'high risk' || rawRisk === 'high' || rawRisk === '2') {
      cls = 2;
    }

    if (cls !== null) {
      validRows.push(r);
      labels.push(cls);
    }
  }

  if (validRows.length < 10) {
    throw new Error('Dataset must contain at least 10 labeled records to train and evaluate models.');
  }

  // If a very large CSV is uploaded (>1500 rows), take a deterministic stratified sample
  // so model training completes responsively without blocking the server.
  let modelRows = validRows;
  let modelLabels = labels;
  if (validRows.length > 1500) {
    const keepRatio = 1500 / validRows.length;
    const sub = stratifiedSplit(validRows, labels, keepRatio, config.randomSeed);
    modelRows = sub.testItems;
    modelLabels = sub.testLabels;
  }

  // Strictly split BEFORE fitting imputer or scaler (zero data leakage)
  const split = stratifiedSplit(modelRows, modelLabels, config.testSize, config.randomSeed);
  const fittedPreprocessor = fitPreprocessorOnRows(split.trainItems, config);

  const XTrain = split.trainItems.map((r) =>
    transformRowToScaledVector(r, fittedPreprocessor)
  );
  const XTest = split.testItems.map((r) =>
    transformRowToScaledVector(r, fittedPreprocessor)
  );
  const yTrain = split.trainLabels;
  const yTest = split.testLabels;

  // 1. Logistic Regression
  const t0 = performance.now();
  const logistic = trainLogisticRegression(XTrain, yTrain, 450, 0.28, 0.015);
  const lrTime = performance.now() - t0;
  const lrProbs = XTest.map((x) => predictProbaLogistic(logistic, x));
  const lrEval = evaluatePredictions(
    'logistic_regression',
    'Logistic Regression',
    'Logistic Reg.',
    'Multinomial softmax linear classifier with L2 regularization, modeling additive log-odds across standardized clinical variables.',
    { solver: 'Batch Gradient Descent', regularization: 'L2 (lambda=0.015)', max_iter: 450 },
    yTest,
    lrProbs,
    logistic.featureImportance,
    XTrain.length,
    lrTime
  );

  // 2. Decision Tree Classifier
  const t1 = performance.now();
  const decisionTree = trainDecisionTree(XTrain, yTrain, 6, 6, 2, config.randomSeed);
  const dtTime = performance.now() - t1;
  const dtProbs = XTest.map((x) => predictProbaTree(decisionTree.root, x));
  const dtEval = evaluatePredictions(
    'decision_tree',
    'Decision Tree Classifier',
    'Decision Tree',
    'CART multi-class decision tree partitioning patient risk space via Gini impurity minimization for interpretable rule splits.',
    { criterion: 'Gini Impurity', max_depth: 6, min_samples_split: 6, min_samples_leaf: 2 },
    yTest,
    dtProbs,
    decisionTree.featureImportance,
    XTrain.length,
    dtTime
  );

  // 3. Random Forest Classifier
  const t2 = performance.now();
  const randomForest = trainRandomForest(XTrain, yTrain, 35, 8, config.randomSeed);
  const rfTime = performance.now() - t2;
  const rfProbs = XTest.map((x) => predictProbaRandomForest(randomForest, x));
  const rfEval = evaluatePredictions(
    'random_forest',
    'Random Forest Classifier',
    'Random Forest',
    'Bagged ensemble of 35 decorrelated CART trees using bootstrap resampling and random feature subsampling to reduce variance.',
    { n_estimators: 35, max_depth: 8, max_features: 'sqrt(p)', bootstrap: 'True' },
    yTest,
    rfProbs,
    randomForest.featureImportance,
    XTrain.length,
    rfTime
  );

  // 4. Gradient Boosting Classifier
  const t3 = performance.now();
  const gradientBoosting = trainGradientBoosting(XTrain, yTrain, 38, 0.12, 3);
  const gbTime = performance.now() - t3;
  const gbProbs = XTest.map((x) => predictProbaGradientBoosting(gradientBoosting, x));
  const gbEval = evaluatePredictions(
    'gradient_boosting',
    'Gradient Boosting Classifier',
    'Grad. Boosting',
    'Stagewise additive multinomial gradient boosted trees iteratively fitting negative log-loss gradients for high predictive accuracy.',
    { n_estimators: 38, learning_rate: 0.12, max_depth: 3, loss: 'Multinomial Deviance' },
    yTest,
    gbProbs,
    gradientBoosting.featureImportance,
    XTrain.length,
    gbTime
  );

  const models: Record<ModelKey, ModelEvaluation> = {
    logistic_regression: lrEval,
    decision_tree: dtEval,
    random_forest: rfEval,
    gradient_boosting: gbEval,
  };

  const orderedKeys: ModelKey[] = [
    'gradient_boosting',
    'random_forest',
    'logistic_regression',
    'decision_tree',
  ];
  let bestModelKey: ModelKey = orderedKeys[0];
  for (const key of orderedKeys) {
    if (
      models[key].f1Macro > models[bestModelKey].f1Macro ||
      (models[key].f1Macro === models[bestModelKey].f1Macro &&
        models[key].accuracy > models[bestModelKey].accuracy)
    ) {
      bestModelKey = key;
    }
  }

  return {
    bundle: {
      trainedAt: new Date().toISOString(),
      datasetName,
      isSynthetic,
      config,
      bestModelKey,
      models,
    },
    fittedPreprocessor,
    logistic,
    decisionTree,
    randomForest,
    gradientBoosting,
  };
}

export function predictPatientRisk(
  runtime: TrainedRuntimeState,
  input: PredictionInput
): PredictionResult {
  const heightM = input.height_cm / 100;
  const bmi = Math.round((input.weight_kg / (heightM * heightM)) * 10) / 10;

  const rawRow: RawDatasetRow = {
    age: input.age,
    gender: input.gender,
    height_cm: input.height_cm,
    weight_kg: input.weight_kg,
    bmi,
    systolic_bp: input.systolic_bp,
    diastolic_bp: input.diastolic_bp,
    glucose_level: input.glucose_level,
    cholesterol_level: input.cholesterol_level,
    heart_rate: input.heart_rate,
    smoking_status: input.smoking_status,
    physical_activity: input.physical_activity,
    family_history: input.family_history,
    existing_conditions: input.existing_conditions,
  };

  const scaledVec = transformRowToScaledVector(rawRow, runtime.fittedPreprocessor);

  let probs: number[];
  if (input.model_key === 'logistic_regression') {
    probs = predictProbaLogistic(runtime.logistic, scaledVec);
  } else if (input.model_key === 'decision_tree') {
    probs = predictProbaTree(runtime.decisionTree.root, scaledVec);
  } else if (input.model_key === 'random_forest') {
    probs = predictProbaRandomForest(runtime.randomForest, scaledVec);
  } else {
    probs = predictProbaGradientBoosting(runtime.gradientBoosting, scaledVec);
  }

  let bestIdx = 0;
  for (let c = 1; c < NUM_CLASSES; c++) {
    if (probs[c] > probs[bestIdx]) bestIdx = c;
  }

  const predicted_risk = RISK_CLASSES[bestIdx];
  const probabilities: Record<RiskCategory, number> = {
    'Low Risk': Math.round(probs[0] * 10000) / 10000,
    'Moderate Risk': Math.round(probs[1] * 10000) / 10000,
    'High Risk': Math.round(probs[2] * 10000) / 10000,
  };

  const modelEval = runtime.bundle.models[input.model_key];

  // Build clinical parameter summary against standard adult reference ranges
  const parameter_summary: PredictionResult['parameter_summary'] = [
    {
      parameter: 'Body Mass Index (BMI)',
      value: `${bmi} kg/m²`,
      reference: '18.5 – 24.9 kg/m²',
      status: bmi >= 30 ? 'Elevated' : bmi >= 25 || bmi < 18.5 ? 'Borderline' : 'Normal',
    },
    {
      parameter: 'Blood Pressure',
      value: `${input.systolic_bp}/${input.diastolic_bp} mmHg`,
      reference: '< 120 / 80 mmHg',
      status:
        input.systolic_bp >= 140 || input.diastolic_bp >= 90
          ? 'Elevated'
          : input.systolic_bp >= 120 || input.diastolic_bp >= 80
            ? 'Borderline'
            : 'Normal',
    },
    {
      parameter: 'Blood Glucose',
      value: `${input.glucose_level} mg/dL`,
      reference: '70 – 99 mg/dL (fasting)',
      status:
        input.glucose_level >= 126
          ? 'Elevated'
          : input.glucose_level >= 100
            ? 'Borderline'
            : 'Normal',
    },
    {
      parameter: 'Total Cholesterol',
      value: `${input.cholesterol_level} mg/dL`,
      reference: '< 200 mg/dL',
      status:
        input.cholesterol_level >= 240
          ? 'Elevated'
          : input.cholesterol_level >= 200
            ? 'Borderline'
            : 'Normal',
    },
    {
      parameter: 'Resting Heart Rate',
      value: `${input.heart_rate} bpm`,
      reference: '60 – 90 bpm',
      status:
        input.heart_rate >= 100 || input.heart_rate < 50
          ? 'Elevated'
          : input.heart_rate > 90
            ? 'Borderline'
            : 'Normal',
    },
    {
      parameter: 'Smoking Status',
      value: input.smoking_status,
      reference: 'Never',
      status:
        input.smoking_status === 'Current'
          ? 'Elevated'
          : input.smoking_status === 'Former'
            ? 'Borderline'
            : 'Normal',
    },
    {
      parameter: 'Physical Activity',
      value: input.physical_activity,
      reference: 'Moderate – High',
      status:
        input.physical_activity === 'Low'
          ? 'Elevated'
          : input.physical_activity === 'Moderate'
            ? 'Borderline'
            : 'Normal',
    },
    {
      parameter: 'Family Medical History',
      value: input.family_history,
      reference: 'None',
      status:
        input.family_history === 'Multiple'
          ? 'Elevated'
          : input.family_history !== 'None'
            ? 'Borderline'
            : 'Normal',
    },
    {
      parameter: 'Existing Conditions',
      value: input.existing_conditions,
      reference: 'None',
      status:
        input.existing_conditions === 'Multiple' ||
        input.existing_conditions === 'Heart Disease' ||
        input.existing_conditions === 'Diabetes' ||
        input.existing_conditions === 'Hypertension'
          ? 'Elevated'
          : input.existing_conditions !== 'None'
            ? 'Borderline'
            : 'Normal',
    },
  ];

  // Combine model feature importance with individual clinical risk elevation
  const featureMeta: Record<
    string,
    { value: string; ref: string; status: 'Elevated' | 'Borderline' | 'Normal'; weight: number }
  > = {
    age: {
      value: `${input.age} yrs`,
      ref: '< 45 yrs baseline',
      status: input.age >= 60 ? 'Elevated' : input.age >= 45 ? 'Borderline' : 'Normal',
      weight: input.age >= 60 ? 1.5 : input.age >= 45 ? 1.15 : 0.7,
    },
    gender: {
      value: input.gender,
      ref: 'Demographic',
      status: 'Normal',
      weight: 0.6,
    },
    bmi: {
      value: `${bmi} kg/m²`,
      ref: '18.5 – 24.9',
      status: bmi >= 30 ? 'Elevated' : bmi >= 25 ? 'Borderline' : 'Normal',
      weight: bmi >= 30 ? 1.5 : bmi >= 25 ? 1.15 : 0.7,
    },
    systolic_bp: {
      value: `${input.systolic_bp} mmHg`,
      ref: '< 120 mmHg',
      status:
        input.systolic_bp >= 140 ? 'Elevated' : input.systolic_bp >= 120 ? 'Borderline' : 'Normal',
      weight: input.systolic_bp >= 140 ? 1.6 : input.systolic_bp >= 120 ? 1.2 : 0.7,
    },
    diastolic_bp: {
      value: `${input.diastolic_bp} mmHg`,
      ref: '< 80 mmHg',
      status:
        input.diastolic_bp >= 90 ? 'Elevated' : input.diastolic_bp >= 80 ? 'Borderline' : 'Normal',
      weight: input.diastolic_bp >= 90 ? 1.5 : input.diastolic_bp >= 80 ? 1.15 : 0.7,
    },
    glucose_level: {
      value: `${input.glucose_level} mg/dL`,
      ref: '70 – 99 mg/dL',
      status:
        input.glucose_level >= 126
          ? 'Elevated'
          : input.glucose_level >= 100
            ? 'Borderline'
            : 'Normal',
      weight: input.glucose_level >= 126 ? 1.6 : input.glucose_level >= 100 ? 1.2 : 0.7,
    },
    cholesterol_level: {
      value: `${input.cholesterol_level} mg/dL`,
      ref: '< 200 mg/dL',
      status:
        input.cholesterol_level >= 240
          ? 'Elevated'
          : input.cholesterol_level >= 200
            ? 'Borderline'
            : 'Normal',
      weight: input.cholesterol_level >= 240 ? 1.5 : input.cholesterol_level >= 200 ? 1.2 : 0.7,
    },
    heart_rate: {
      value: `${input.heart_rate} bpm`,
      ref: '60 – 90 bpm',
      status: input.heart_rate >= 100 ? 'Elevated' : input.heart_rate > 90 ? 'Borderline' : 'Normal',
      weight: input.heart_rate >= 100 ? 1.4 : input.heart_rate > 90 ? 1.1 : 0.7,
    },
    smoking_status: {
      value: input.smoking_status,
      ref: 'Never',
      status:
        input.smoking_status === 'Current'
          ? 'Elevated'
          : input.smoking_status === 'Former'
            ? 'Borderline'
            : 'Normal',
      weight: input.smoking_status === 'Current' ? 1.6 : input.smoking_status === 'Former' ? 1.1 : 0.6,
    },
    physical_activity: {
      value: input.physical_activity,
      ref: 'Moderate – High',
      status:
        input.physical_activity === 'Low'
          ? 'Elevated'
          : input.physical_activity === 'Moderate'
            ? 'Borderline'
            : 'Normal',
      weight: input.physical_activity === 'Low' ? 1.4 : 0.8,
    },
    family_history: {
      value: input.family_history,
      ref: 'None',
      status:
        input.family_history === 'Multiple'
          ? 'Elevated'
          : input.family_history !== 'None'
            ? 'Borderline'
            : 'Normal',
      weight: input.family_history === 'Multiple' ? 1.5 : input.family_history !== 'None' ? 1.2 : 0.6,
    },
    existing_conditions: {
      value: input.existing_conditions,
      ref: 'None',
      status:
        input.existing_conditions === 'Multiple' ||
        input.existing_conditions === 'Heart Disease' ||
        input.existing_conditions === 'Diabetes' ||
        input.existing_conditions === 'Hypertension'
          ? 'Elevated'
          : input.existing_conditions !== 'None'
            ? 'Borderline'
            : 'Normal',
      weight:
        ENCODING_MAPS.existing_conditions[input.existing_conditions] >= 2 ? 1.6 : 0.7,
    },
  };

  const contributing_factors: ContributingFactor[] = modelEval.featureImportance
    .map((fi) => {
      const meta = featureMeta[fi.feature];
      const score = Math.round(fi.importance * (meta?.weight ?? 1) * 1000) / 10;
      return {
        feature: fi.feature,
        label: fi.label,
        patientValue: meta?.value ?? '',
        referenceRange: meta?.ref ?? '',
        status: meta?.status ?? 'Normal',
        contributionScore: score,
      };
    })
    .sort((a, b) => b.contributionScore - a.contributionScore)
    .slice(0, 6);

  // Generate tailored, non-diagnostic general educational wellness recommendations
  const recommendations: string[] = [];
  if (input.systolic_bp >= 130 || input.diastolic_bp >= 85) {
    recommendations.push(
      'Blood pressure readings are above optimal range; consider regular home blood pressure monitoring, dietary sodium reduction, and consultation with a licensed clinician.'
    );
  }
  if (input.glucose_level >= 110) {
    recommendations.push(
      'Elevated blood glucose detected; balanced low-glycemic nutrition, post-meal walking, and routine HbA1c screening with a healthcare provider may be helpful.'
    );
  }
  if (input.cholesterol_level >= 200) {
    recommendations.push(
      'Total cholesterol is borderline or elevated; increasing soluble fiber intake, limiting saturated fats, and reviewing a full lipid panel with a physician is recommended.'
    );
  }
  if (bmi >= 25) {
    recommendations.push(
      'BMI is above the standard reference band (18.5–24.9 kg/m²); gradual, sustainable nutrition and activity adjustments can reduce cardiometabolic strain.'
    );
  }
  if (input.smoking_status === 'Current') {
    recommendations.push(
      'Active smoking substantially amplifies cardiovascular risk; evidence-based cessation counseling and support programs yield rapid risk reductions.'
    );
  }
  if (input.physical_activity === 'Low') {
    recommendations.push(
      'Physical activity is currently low; aiming gradually toward 150 minutes/week of moderate aerobic exercise supports blood pressure, glucose, and lipid control.'
    );
  }
  if (predicted_risk === 'High Risk') {
    recommendations.push(
      'Because multiple risk parameters co-occur in the High Risk tier, scheduling a comprehensive preventive checkup with a qualified medical professional is advised.'
    );
  } else if (recommendations.length === 0) {
    recommendations.push(
      'Current health parameters fall largely within standard reference ranges; maintain regular physical activity, balanced nutrition, and routine annual wellness checkups.'
    );
  }

  return {
    patient_id: input.patient_id?.trim() || `EVAL-${Date.now().toString().slice(-5)}`,
    model_key: input.model_key,
    model_name: modelEval.name,
    predicted_risk,
    probabilities,
    confidence: probabilities[predicted_risk],
    bmi,
    contributing_factors,
    parameter_summary,
    recommendations,
    target_definition_note:
      'Target classes (Low Risk = 0, Moderate Risk = 1, High Risk = 2) are predicted directly by the trained 3-class supervised classifier via argmax over class probabilities, without arbitrary binary probability cutoffs.',
    disclaimer:
      'Educational Decision-Support Notice: This prediction is generated by a machine learning model trained for educational demonstration and is NOT a medical diagnosis or substitute for licensed clinical evaluation.',
    created_at: new Date().toISOString(),
  };
}
