import { useOnlineStatus } from '../hooks/useOnlineStatus';

export const OfflineBanner = () => {
  const isOnline = useOnlineStatus();

  if (isOnline) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        background: '#dc3545',
        color: 'white',
        padding: '12px',
        textAlign: 'center',
        fontSize: '14px',
        fontWeight: 'bold',
        zIndex: 9999,
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
      }}
    >
      You're offline. Some features may not be available.
    </div>
  );
};
