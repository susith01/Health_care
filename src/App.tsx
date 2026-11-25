import React, { useState } from 'react';
import {
  Activity,
  BarChart3,
  BookOpen,
  Cpu,
  Database,
  FileText,
  Gauge,
  LayoutDashboard,
  Menu,
  SlidersHorizontal,
  Stethoscope,
  UserPlus,
  X,
} from 'lucide-react';
import { AboutProject } from './pages/AboutProject';
import { DataPreprocessing } from './pages/DataPreprocessing';
import { ExploratoryAnalysis } from './pages/ExploratoryAnalysis';
import { HomeDashboard, type NavPageId } from './pages/HomeDashboard';
import { MachineLearningModels } from './pages/MachineLearningModels';
import { ModelPerformance } from './pages/ModelPerformance';
import { PatientData } from './pages/PatientData';
import { PatientRegistration } from './pages/PatientRegistration';
import { ReportsPage } from './pages/ReportsPage';
import { RiskPrediction } from './pages/RiskPrediction';

interface NavItem {
  id: NavPageId;
  label: string;
  section: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'home', label: 'Home Dashboard', section: 'Overview', icon: LayoutDashboard },
  { id: 'register', label: 'Patient Registration', section: 'Clinical Records', icon: UserPlus },
  { id: 'patients', label: 'Patient Data', section: 'Clinical Records', icon: Database },
  { id: 'preprocessing', label: 'Data Preprocessing', section: 'Analytics & ML', icon: SlidersHorizontal },
  { id: 'eda', label: 'Exploratory Data Analysis', section: 'Analytics & ML', icon: BarChart3 },
  { id: 'models', label: 'Machine Learning Models', section: 'Analytics & ML', icon: Cpu },
  { id: 'predict', label: 'Risk Prediction', section: 'Decision Support', icon: Activity },
  { id: 'performance', label: 'Model Performance', section: 'Decision Support', icon: Gauge },
  { id: 'reports', label: 'Reports', section: 'Decision Support', icon: FileText },
  { id: 'about', label: 'About Project', section: 'Documentation', icon: BookOpen },
];

export default function App() {
  const [activePage, setActivePage] = useState<NavPageId>('home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const activeNav = NAV_ITEMS.find((n) => n.id === activePage) ?? NAV_ITEMS[0];

  const handleNavClick = (id: NavPageId) => {
    setActivePage(id);
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior });
  };

  const renderPage = () => {
    switch (activePage) {
      case 'home':
        return <HomeDashboard onNavigate={handleNavClick} />;
      case 'register':
        return <PatientRegistration />;
      case 'patients':
        return <PatientData />;
      case 'preprocessing':
        return <DataPreprocessing />;
      case 'eda':
        return <ExploratoryAnalysis />;
      case 'models':
        return <MachineLearningModels />;
      case 'predict':
        return <RiskPrediction />;
      case 'performance':
        return <ModelPerformance />;
      case 'reports':
        return <ReportsPage />;
      case 'about':
        return <AboutProject />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:flex-col w-64 shrink-0 bg-white border-r border-slate-200 sticky top-0 h-screen">
        <div className="px-5 h-16 border-b border-slate-200 flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-md bg-sky-700 text-white flex items-center justify-center shrink-0">
            <Stethoscope className="w-4 h-4" />
          </div>
          <span className="font-bold text-sm tracking-tight text-slate-900 truncate">
            HealthRisk Analytics
          </span>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1" aria-label="Dashboard Navigation">
          {NAV_ITEMS.map((item, idx) => {
            const Icon = item.icon;
            const isActive = item.id === activePage;
            const showGroup = idx === 0 || NAV_ITEMS[idx - 1].section !== item.section;

            return (
              <React.Fragment key={item.id}>
                {showGroup && (
                  <div className="px-2.5 pt-3 pb-1 text-[11px] font-semibold text-slate-400 first:pt-0">
                    {item.section}
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => handleNavClick(item.id)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                    isActive
                      ? 'bg-sky-50 text-sky-800 font-semibold'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  <Icon
                    className={`w-4 h-4 shrink-0 ${isActive ? 'text-sky-700' : 'text-slate-400'}`}
                  />
                  <span className="truncate">{item.label}</span>
                </button>
              </React.Fragment>
            );
          })}
        </nav>

        <div className="px-5 py-3.5 border-t border-slate-200 text-[11px] text-slate-500">
          Educational Decision Support · Not for clinical diagnosis
        </div>
      </aside>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/40"
            onClick={() => setMobileMenuOpen(false)}
          />
          <aside className="relative w-72 max-w-[85vw] bg-white h-full flex flex-col border-r border-slate-200 z-10">
            <div className="px-4 h-14 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-md bg-sky-700 text-white flex items-center justify-center">
                  <Stethoscope className="w-4 h-4" />
                </div>
                <span className="font-bold text-sm text-slate-900">HealthRisk Analytics</span>
              </div>
              <button
                type="button"
                onClick={() => setMobileMenuOpen(false)}
                className="p-2 text-slate-500 hover:text-slate-800"
                aria-label="Close navigation menu"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-1">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const isActive = item.id === activePage;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleNavClick(item.id)}
                    className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-sky-50 text-sky-800 font-semibold'
                        : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </aside>
        </div>
      )}

      {/* Main Workspace */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Top Bar Contract: Breadcrumb on left + 1 Primary Action on right */}
        <header className="sticky top-0 z-30 h-16 bg-white/95 backdrop-blur-xs border-b border-slate-200 px-4 sm:px-8 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileMenuOpen(true)}
              className="lg:hidden p-2 -ml-2 text-slate-600 hover:text-slate-900"
              aria-label="Open navigation menu"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2 text-xs sm:text-sm min-w-0 truncate">
              <span className="text-slate-400 hidden sm:inline whitespace-nowrap">
                {activeNav.section}
              </span>
              <span className="text-slate-300 hidden sm:inline" aria-hidden="true">
                /
              </span>
              <span className="font-semibold text-slate-900 truncate">{activeNav.label}</span>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <button
              type="button"
              onClick={() => handleNavClick('predict')}
              className="px-3.5 py-2 text-xs font-medium text-white bg-sky-700 rounded-md hover:bg-sky-800 transition-colors whitespace-nowrap shrink-0"
            >
              Predict Risk
            </button>
          </div>
        </header>

        <main className="flex-1 p-4 sm:p-8 max-w-[1360px] w-full mx-auto">
          {renderPage()}
        </main>
      </div>
    </div>
  );
}
