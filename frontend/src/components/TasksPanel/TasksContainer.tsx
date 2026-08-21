import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useActiveTasks } from '../../hooks/useActiveTasks';
import type { TaskStatus } from '../../types';
import { TasksIndicator } from './TasksPanel';
import { TasksPanel } from './TasksPanel';

export const projectGlobalTasks = (tasks: TaskStatus[]) => {
  const visibleTasks = tasks.filter((task) => task.task_type !== 'reset_dub');
  const activeCount = visibleTasks.filter(
    (task) =>
      task.status === 'queued'
      || task.status === 'running'
      || task.status === 'awaiting_subtitle_review',
  ).length;

  return {
    activeTasks: visibleTasks,
    activeCount,
    hasActive: activeCount > 0,
  };
};

/**
 * Connected component that wires useActiveTasks data to the TasksIndicator
 * and TasksPanel UI. Must be rendered inside BrowserRouter so useNavigate works.
 */
export const TasksContainer = () => {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();
  const taskState = useActiveTasks();
  const { activeTasks, activeCount, hasActive } = projectGlobalTasks(taskState.activeTasks);

  const handleViewJob = (taskId: string) => {
    setIsOpen(false);
    navigate(`/job/${taskId}`);
  };

  if (!hasActive) return null;

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
