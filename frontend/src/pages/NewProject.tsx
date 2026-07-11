import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCreateProject } from '../hooks/useProjects';
import { useFileBrowser } from '../hooks/useFileBrowser';
import { useSettings } from '../hooks/useSettings';
import { useUIStore } from '../stores/uiStore';
import { ProjectCreation } from '../components/ProjectCreation';
import styles from './NewProject.module.css';

export const NewProject = () => {
  const navigate = useNavigate();
  const createProject = useCreateProject();
  const setCurrentProjectId = useUIStore((state) => state.setCurrentProjectId);
  const { settings } = useSettings();

  // Start the browser at projects_root_path if configured, otherwise filesystem root
  const startPath = settings?.projects_root_path || '/';
  const browser = useFileBrowser(startPath);

  useEffect(() => {
    browser.navigate(startPath);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startPath]);

  const handleCreate = async (path: string) => {
    try {
      const project = await createProject.mutateAsync({ path });
      setCurrentProjectId(project.id);
      navigate(`/project/${project.id}`);
    } catch (err) {
      console.error('Failed to create project:', err);
    }
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <button
          className={styles.backButton}
          onClick={() => navigate('/')}
          type="button"
        >
          ← Back
        </button>
        <h1 className={styles.title}>New Project</h1>
      </header>

      <main className={styles.content}>
        {createProject.isError && (
          <div className={styles.errorBanner}>
            Failed to create project: {(createProject.error as Error).message}
          </div>
        )}

        <ProjectCreation
          initialPath={browser.currentPath}
          nodes={browser.nodes}
          isLoading={browser.isLoading || createProject.isPending}
          onLoadDirectory={browser.navigate}
          onCreateProject={handleCreate}
          onCancel={() => navigate('/')}
        />
      </main>
    </div>
  );
};
