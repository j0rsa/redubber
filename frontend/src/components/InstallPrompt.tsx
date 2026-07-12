import { useState, useEffect } from 'react';

export const InstallPrompt = () => {
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    const handler = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e);

      // Show prompt after 2 visits
      const visits = parseInt(localStorage.getItem('visits') || '0');
      if (visits >= 2) {
        setShowPrompt(true);
      }
      localStorage.setItem('visits', String(visits + 1));
    };

    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;

    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;

    if (outcome === 'accepted') {
      setShowPrompt(false);
    }
    setDeferredPrompt(null);
  };

  if (!showPrompt) return null;

  return (
    <div className="install-prompt" style={{
      position: 'fixed',
      bottom: '20px',
      left: '50%',
      transform: 'translateX(-50%)',
      background: '#1976d2',
      color: 'white',
      padding: '16px 24px',
      borderRadius: '8px',
      boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      zIndex: 1000
    }}>
      <p style={{ margin: 0, fontSize: '14px' }}>Install Redubber for quick access</p>
      <button
        onClick={handleInstall}
        style={{
          background: 'white',
          color: '#1976d2',
          border: 'none',
          padding: '8px 16px',
          borderRadius: '4px',
          cursor: 'pointer',
          fontWeight: 600
        }}
      >
        Install
      </button>
      <button
        onClick={() => setShowPrompt(false)}
        style={{
          background: 'transparent',
          color: 'white',
          border: '1px solid white',
          padding: '8px 16px',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        Not now
      </button>
    </div>
  );
};
