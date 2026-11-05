import React, { Component } from 'react';

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("UI Uncaught Error caught by ErrorBoundary:", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '2rem', background: '#1e1e2f', color: '#ff6b6b', fontFamily: 'sans-serif', minHeight: '100vh' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>⚠️ UI Runtime Error Encountered</h2>
          <pre style={{ background: '#0f0f18', padding: '1rem', borderRadius: '8px', color: '#f8f8f2', overflow: 'auto', marginTop: '1rem' }}>
            {this.state.error?.toString()}
          </pre>
          <details style={{ marginTop: '1rem', color: '#a0a0b0' }}>
            <summary style={{ cursor: 'pointer' }}>Component Stack Trace</summary>
            <pre style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
              {this.state.errorInfo?.componentStack}
            </pre>
          </details>
          <button 
            onClick={() => window.location.reload()}
            style={{ marginTop: '1.5rem', padding: '0.5rem 1rem', background: '#3b82f6', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
          >
            Reload Application
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
