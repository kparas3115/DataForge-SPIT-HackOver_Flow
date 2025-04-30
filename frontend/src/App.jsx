import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";
import LoginPage from "./pages/login";
import SignupPage from "./pages/signup";
import LandingPage from "./pages/landing";
import ResearchPapers from "./pages/research-papers";
import PaperDetails from "./pages/paper-details";
import Datasets from "./pages/datasets";
import DatasetUpload from "./pages/dataset-upload";
import DatasetDetail from "./pages/DatasetDetail";
import DashboardLayout from "./components/layout/DashboardLayout";
import AuthenticatedLayout from "./components/layout/authenticated-layout";
import Articles from "./pages/articles"; 
import ProfileForm from "./pages/ProfileForm";
import UpdateProfile from "./pages/update-profile";
import News from "./pages/news";
import Connections from "./pages/connections";
import Messages from "./pages/messages";
import ConnectionsList from "./pages/ConnectionsList";
import { AuthProvider } from '@/contexts/AuthContext';
import JournalRecommender from "./components/JournalRecommender";
import ResearchChatbot from "./components/ResearchChatbot";

const router = createBrowserRouter([
    // Public Routes
    {
        path: "/",
        element: <LandingPage />,
    },
    {
        path: "/login",
        element: <LoginPage />,
    },
    {
        path: "/signup",
        element: <SignupPage />,
    },

    // Dashboard Routes wrapped in DashboardLayout
    {
        path: "/dashboard",
        element: <DashboardLayout />,
        children: [
            {
                path: "connections",
                element: <Connections />,
            },
            // Redirect /dashboard to /dashboard/connections
            {
                path: "",
                element: <Navigate to="connections" />,
            },
        ],
    },

    // Authenticated Routes wrapped in AuthenticatedLayout
    {
        element: <AuthenticatedLayout />,
        children: [
            {
                path: "/recommender",
                element: <JournalRecommender />,
            },
            {
                path: "/chatbot",
                element: <ResearchChatbot />,
            },
            {
                path: "/research-papers",
                element: <ResearchPapers />,
            },
            {
                path: "/research-papers/:id",
                element: <PaperDetails />,
            },
            {
                path: "/datasets",
                element: <Datasets />,
            },
            {
                path: "/dataset-upload",
                element: <DatasetUpload />,
            },
            {
                path: "/news",
                element: <News />,
            },
            {
                path: "/dataset/:id",
                element: <DatasetDetail />,
            },
            {
                path: "/update-profile",
                element: <ProfileForm />,
            },
            {
                path: "/articles",
                element: <Articles />,
            },
            {
                path: "/news",
                element: <News />,
            },
            {
                path: "/connections",
                element: <Connections />,
            },
            { path: "/my-connections", element: <ConnectionsList /> },
            { path: "/messages/:userId", element: <Messages /> },
            {
                path: "/messages",
                element: <Messages />,
            },
        ],
    },

    // Redirect unmatched routes
    {
        path: "*",
        element: <Navigate to="/" />,
    },
]);

function App() {
  return (
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  );
}

export default App;