import React from 'react';

export default function Layout({ children }) {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="bg-blue-800 text-white p-4">
        <div className="container mx-auto">
          <h1 className="text-2xl font-bold">Economic Risk Index (ERI)</h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow container mx-auto p-4">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-gray-200 p-4 text-center">
        <p>© 2025 FinancialCrisisML. All rights reserved.</p>
      </footer>
    </div>
  );
}
