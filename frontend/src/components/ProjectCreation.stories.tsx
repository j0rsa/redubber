import type { Meta, StoryObj } from '@storybook/react-vite';
import { ProjectCreation } from './ProjectCreation';
import type { FileNode } from './FileBrowser';

const meta: Meta<typeof ProjectCreation> = {
  title: 'Components/ProjectCreation',
  component: ProjectCreation,
  decorators: [
    (Story) => (
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', width: '100%', background: 'var(--color-bg-secondary)' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: 'fullscreen',
    viewport: {
      defaultViewport: 'desktop',
    },
    backgrounds: {
      default: 'light-gray',
    },
    docs: {
      description: {
        component: 'Project creation file browser panel. Used as the right-hand panel in ProjectHub.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof ProjectCreation>;

// Mock data generators
const createDirectory = (name: string, path: string, children?: FileNode[]): FileNode => ({
  name,
  path,
  type: 'directory',
  children,
});

const createFile = (name: string, path: string, size: number): FileNode => ({
  name,
  path,
  type: 'file',
  size,
});

// Home directory with common folders
const homeDirectoryNodes: FileNode[] = [
  createDirectory('Documents', '/Users/john/Documents', [
    createDirectory('Work', '/Users/john/Documents/Work'),
    createDirectory('Personal', '/Users/john/Documents/Personal'),
    createFile('notes.txt', '/Users/john/Documents/notes.txt', 2048),
  ]),
  createDirectory('Videos', '/Users/john/Videos', [
    createDirectory('Tutorials', '/Users/john/Videos/Tutorials'),
    createDirectory('Meetings', '/Users/john/Videos/Meetings'),
    createDirectory('Webinars', '/Users/john/Videos/Webinars'),
  ]),
  createDirectory('Downloads', '/Users/john/Downloads', [
    createFile('installer.dmg', '/Users/john/Downloads/installer.dmg', 524288000),
    createFile('document.pdf', '/Users/john/Downloads/document.pdf', 2048000),
  ]),
  createDirectory('Desktop', '/Users/john/Desktop'),
  createDirectory('Pictures', '/Users/john/Pictures'),
];

// Videos folder with multiple subfolders
const videosDirectoryNodes: FileNode[] = [
  createDirectory('Tutorials', '/Users/john/Videos/Tutorials', [
    createFile('intro.mp4', '/Users/john/Videos/Tutorials/intro.mp4', 157286400),
    createFile('lesson1.mp4', '/Users/john/Videos/Tutorials/lesson1.mp4', 314572800),
    createFile('lesson2.mp4', '/Users/john/Videos/Tutorials/lesson2.mp4', 471859200),
  ]),
  createDirectory('Meetings', '/Users/john/Videos/Meetings', [
    createFile('standup_2024_01_15.mp4', '/Users/john/Videos/Meetings/standup_2024_01_15.mp4', 524288000),
    createFile('quarterly_review.mp4', '/Users/john/Videos/Meetings/quarterly_review.mp4', 1073741824),
  ]),
  createDirectory('Webinars', '/Users/john/Videos/Webinars', [
    createFile('webinar_part1.mp4', '/Users/john/Videos/Webinars/webinar_part1.mp4', 629145600),
    createFile('webinar_part2.mp4', '/Users/john/Videos/Webinars/webinar_part2.mp4', 629145600),
    createFile('webinar_part3.mp4', '/Users/john/Videos/Webinars/webinar_part3.mp4', 629145600),
  ]),
  createDirectory('Archive', '/Users/john/Videos/Archive'),
];

// Single folder with videos (Tutorials expanded)
const tutorialsFolderNodes: FileNode[] = [
  createFile('intro.mp4', '/Users/john/Videos/Tutorials/intro.mp4', 157286400),
  createFile('lesson1.mp4', '/Users/john/Videos/Tutorials/lesson1.mp4', 314572800),
  createFile('lesson2.mp4', '/Users/john/Videos/Tutorials/lesson2.mp4', 471859200),
  createFile('lesson3.mkv', '/Users/john/Videos/Tutorials/lesson3.mkv', 524288000),
  createFile('final_exam.mp4', '/Users/john/Videos/Tutorials/final_exam.mp4', 1073741824),
  createDirectory('Archived', '/Users/john/Videos/Tutorials/Archived'),
];

// Large nested structure
const largeDirectoryNodes: FileNode[] = [
  createDirectory('2024', '/Projects/Videos/2024', [
    createDirectory('January', '/Projects/Videos/2024/January', [
      createDirectory('Week1', '/Projects/Videos/2024/January/Week1'),
      createDirectory('Week2', '/Projects/Videos/2024/January/Week2'),
      createDirectory('Week3', '/Projects/Videos/2024/January/Week3'),
      createDirectory('Week4', '/Projects/Videos/2024/January/Week4'),
    ]),
    createDirectory('February', '/Projects/Videos/2024/February'),
    createDirectory('March', '/Projects/Videos/2024/March'),
  ]),
  createDirectory('2023', '/Projects/Videos/2023'),
  createDirectory('2022', '/Projects/Videos/2022'),
];

// Empty directory
const emptyDirectoryNodes: FileNode[] = [];

// Default: Home directory view
export const Default: Story = {
  args: {
    initialPath: '/Users/john',
    nodes: homeDirectoryNodes,
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Videos directory with multiple folders
export const VideosDirectory: Story = {
  args: {
    initialPath: '/Users/john/Videos',
    nodes: videosDirectoryNodes,
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Single folder with video files
export const FolderWithVideos: Story = {
  args: {
    initialPath: '/Users/john/Videos/Tutorials',
    nodes: tutorialsFolderNodes,
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Folder selected with auto-filled name
export const FolderSelected: Story = {
  args: {
    initialPath: '/Users/john/Videos',
    nodes: videosDirectoryNodes,
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
  play: async ({ canvasElement }) => {
    // Simulate clicking on "Tutorials" folder
    const tutorials = canvasElement.querySelector('[data-testid="Tutorials"]');
    if (tutorials) {
      (tutorials as HTMLElement).click();
    }
  },
};

// Deep nested directory structure
export const DeepNestedStructure: Story = {
  args: {
    initialPath: '/Projects/Videos',
    nodes: largeDirectoryNodes,
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Empty directory (no files)
export const EmptyDirectory: Story = {
  args: {
    initialPath: '/Users/john/EmptyFolder',
    nodes: emptyDirectoryNodes,
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Loading state
export const Loading: Story = {
  args: {
    initialPath: '/Users/john/Videos',
    nodes: [],
    isLoading: true,
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Root directory (can't navigate up)
export const RootDirectory: Story = {
  args: {
    initialPath: '/',
    nodes: [
      createDirectory('Users', '/Users'),
      createDirectory('Applications', '/Applications'),
      createDirectory('System', '/System'),
      createDirectory('Library', '/Library'),
    ],
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Long folder names (overflow test)
export const LongFolderNames: Story = {
  args: {
    initialPath: '/Projects/Very_Long_Project_Names',
    nodes: [
      createDirectory(
        'This_Is_A_Very_Long_Folder_Name_That_Should_Be_Truncated_In_The_UI',
        '/Projects/Very_Long_Project_Names/This_Is_A_Very_Long_Folder_Name_That_Should_Be_Truncated_In_The_UI'
      ),
      createDirectory(
        'Another_Extremely_Long_Folder_Name_For_Testing_UI_Overflow',
        '/Projects/Very_Long_Project_Names/Another_Extremely_Long_Folder_Name_For_Testing_UI_Overflow'
      ),
      createFile(
        'video_with_a_very_long_filename_that_exceeds_normal_length.mp4',
        '/Projects/Very_Long_Project_Names/video_with_a_very_long_filename_that_exceeds_normal_length.mp4',
        524288000
      ),
    ],
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Many files (scrolling test)
export const ManyFiles: Story = {
  args: {
    initialPath: '/Archive/Videos',
    nodes: Array.from({ length: 50 }, (_, i) =>
      i % 3 === 0
        ? createDirectory(`Folder_${String(i + 1).padStart(3, '0')}`, `/Archive/Videos/Folder_${String(i + 1).padStart(3, '0')}`)
        : createFile(`video_${String(i + 1).padStart(3, '0')}.mp4`, `/Archive/Videos/video_${i + 1}.mp4`, Math.random() * 1073741824)
    ),
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Mixed file types (videos and other files)
export const MixedFileTypes: Story = {
  args: {
    initialPath: '/Users/john/Projects/VideoEditing',
    nodes: [
      createFile('project_file.prproj', '/Users/john/Projects/VideoEditing/project_file.prproj', 10485760),
      createFile('raw_footage.mp4', '/Users/john/Projects/VideoEditing/raw_footage.mp4', 2147483648),
      createFile('edited_video.mp4', '/Users/john/Projects/VideoEditing/edited_video.mp4', 1073741824),
      createFile('audio_track.wav', '/Users/john/Projects/VideoEditing/audio_track.wav', 52428800),
      createFile('thumbnail.png', '/Users/john/Projects/VideoEditing/thumbnail.png', 2097152),
      createDirectory('Assets', '/Users/john/Projects/VideoEditing/Assets'),
      createDirectory('Exports', '/Users/john/Projects/VideoEditing/Exports'),
    ],
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    onCancel: () => console.log('Cancel clicked'),
  },
};

// Without cancel button
export const WithoutCancelButton: Story = {
  args: {
    initialPath: '/Users/john/Videos',
    nodes: videosDirectoryNodes,
    onLoadDirectory: (path) => console.log('Load directory:', path),
    onCreateProject: (path, name) => console.log('Create project:', { path, name }),
    // No onCancel prop
  },
};
