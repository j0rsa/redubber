import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useActiveTasks } from '../../hooks/useActiveTasks';
import { TasksIndicator } from './TasksPanel';
import { TasksPanel } from './TasksPanel';

/**
 * Connected component that wires useActiveTasks data to the TasksIndicator
 * and TasksPanel UI. Must be rendered inside BrowserRouter so useNavigate works.
 */
export const TasksContainer = () => {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();
  const { activeTasks, activeCount } = useActiveTasks();

  const handleViewJob = (taskId: string) => {
    setIsOpen(false);
    navigate(`/job/${taskId}`);
  };

  return (
    <>
      <TasksIndicator
        activeCount={activeCount}
        isOpen={isOpen}
        onClick={() => setIsOpen((prev) => !prev)}
      />
      <TasksPanel
        tasks={activeTasks}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        onViewJob={handleViewJob}
      />
    </>
  );
};
