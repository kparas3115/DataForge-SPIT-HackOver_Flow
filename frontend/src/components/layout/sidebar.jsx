import React, { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { Menu, X, ArrowLeftFromLineIcon, MessageSquare, ChevronDown, ChevronUp } from "lucide-react";
import ChatbotComponent from "../ChatbotComponent";

const Sidebar = ({ user, isCollapsed, toggleSidebar }) => {
  const navigate = useNavigate();
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);

  const handleLogout = () => {
    if (typeof window !== "undefined" && localStorage.getItem("token")) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      navigate("/login");
    }
  };

  const toggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  return (
    <div
      className={`h-screen bg-white text-gray-800 shadow-lg fixed flex flex-col justify-between border-r border-gray-200 transition-all duration-300 ${
        isCollapsed ? "w-16" : "w-64"
      }`}
    >
      <div className="flex flex-col h-full">
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          {!isCollapsed && (
            <div className="flex flex-col">
              <a href="/" className="text-2xl font-bold text-black">ReSync</a>
              {user && (
                <p className="text-sm text-gray-600 mt-2">
                  Hi, {user.firstName} {user.lastName}
                </p>
              )}
            </div>
          )}
          <button
            onClick={toggleSidebar}
            className="p-1 rounded hover:bg-gray-200 focus:outline-none"
          >
            {isCollapsed ? <Menu size={20} /> : <ArrowLeftFromLineIcon size={20} />}
          </button>
        </div>
        
        <nav className="mt-4 flex-grow">
        <NavLink
            to="/recommender"
            className={({ isActive }) =>
              `block py-2.5 px-4 rounded transition duration-200 ${
                isActive ? "bg-black/30 text-blue-600 text-gray-700" : "hover:bg-black/10 text-black"
              } ${isCollapsed ? "text-center" : ""}`
            }
            title={isCollapsed ? "Journal Recommender" : ""}
          >
            {isCollapsed ? "👤" : "Journal Recommender"}
          </NavLink>
          <NavLink
            to="/chatbot"
            className={({ isActive }) =>
              `block py-2.5 px-4 rounded transition duration-200 ${
                isActive ? "bg-black/30 text-blue-600 text-gray-700" : "hover:bg-black/10 text-black"
              } ${isCollapsed ? "text-center" : ""}`
            }
            title={isCollapsed ? "ResearchBuddy" : ""}
          >
            {isCollapsed ? "👤" : "ResearchBuddy"}
          </NavLink>
          <NavLink
            to="/research-papers"
            className={({ isActive }) =>
              `block py-2.5 px-4 rounded transition duration-200 ${
                isActive ? "bg-black/30 text-blue-600 text-gray-700" : "hover:bg-black/10 text-black"
              } ${isCollapsed ? "text-center" : ""}`
            }
            title={isCollapsed ? "Research Papers" : ""}
          >
            {isCollapsed ? "📄" : "Research Papers"}
          </NavLink>
          <NavLink
            to="/datasets"
            className={({ isActive }) =>
              `block py-2.5 px-4 rounded transition duration-200 ${
                isActive ? "bg-black/30 text-blue-600 text-gray-700" : "hover:bg-black/10 text-black"
              } ${isCollapsed ? "text-center" : ""}`
            }
            title={isCollapsed ? "Datasets" : ""}
          >
            {isCollapsed ? "📊" : "Datasets"}
          </NavLink>
          <NavLink
            to="/connections"
            className={({ isActive }) =>
              `block py-2.5 px-4 rounded transition duration-200 ${
                isActive ? "bg-black/30 text-blue-600 text-gray-700" : "hover:bg-black/10 text-black"
              } ${isCollapsed ? "text-center" : ""}`
            }
            title={isCollapsed ? "Connect" : ""}
          >
            {isCollapsed ? "🤝" : "Connect"}
          </NavLink>
          <NavLink
            to="/news"
            className={({ isActive }) =>
              `block py-2.5 px-4 rounded transition duration-200 ${
                isActive ? "bg-black/30 text-blue-600 text-gray-700" : "hover:bg-black/10 text-black"
              } ${isCollapsed ? "text-center" : ""}`
            }
            title={isCollapsed ? "News" : ""}
          >
            {isCollapsed ? "📰" : "News"}
          </NavLink>
          <NavLink
            to="/update-profile"
            className={({ isActive }) =>
              `block py-2.5 px-4 rounded transition duration-200 ${
                isActive ? "bg-black/30 text-blue-600 text-gray-700" : "hover:bg-black/10 text-black"
              } ${isCollapsed ? "text-center" : ""}`
            }
            title={isCollapsed ? "Profile" : ""}
          >
            {isCollapsed ? "👤" : "Profile"}
          </NavLink>
          
          {/* Chatbot Section */}
          {/* <div className="mt-4 border-t border-gray-200"> */}
            {/* <button
              onClick={toggleChatbot}
              className={`flex items-center w-full py-2.5 px-4 rounded transition duration-200 hover:bg-black/10 ${
                isChatbotOpen ? "bg-black/5" : ""
              } ${isCollapsed ? "justify-center" : "justify-between"}`}
              title={isCollapsed ? "Research Assistant" : ""}
            >
              {isCollapsed ? (
                <MessageSquare size={18} />
              ) : (
                <>
                  <div className="flex items-center">
                    <MessageSquare size={18} className="mr-2" />
                    <span>Research Assistant</span>
                  </div>
                  {isChatbotOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </>
              )}
            </button> */}
            
            {/* Chatbot Component - Only show when not collapsed and chatbot is open */}
            {/* {!isCollapsed && isChatbotOpen && (
              <div className="p-2 h-96 border-t border-gray-100">
                <ChatbotComponent />
              </div>
            )} */}
          {/* </div> */}
        </nav>
        
        <div className="p-4 border-t border-gray-200 mt-auto">
          <button
            onClick={handleLogout}
            className={`w-full py-2.5 rounded transition duration-200 bg-red-500 hover:bg-red-600 text-white ${
              isCollapsed ? "px-2 text-sm" : "px-4"
            }`}
            title={isCollapsed ? "Logout" : ""}
          >
            {isCollapsed ? "🚪" : "Logout"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;