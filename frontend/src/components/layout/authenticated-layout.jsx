import React, { useState, useEffect } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import Sidebar from "./Sidebar";
import axios from "axios";
import AiAssistant from "../../AiAssistant";

const AuthenticatedLayout = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const isAuthenticated = !!localStorage.getItem("token");

  const fetchUserData = async (token) => {
    try {
      const res = await axios.get("http://localhost:3000/api/v1/users/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      return res.data.user;
    } catch (err) {
      console.error("Error fetching user data:", err.response?.data || err.message);
      return null;
    }
  };

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  useEffect(() => {
    const loadUserData = async () => {
      const storedUser = localStorage.getItem("user");
      const token = localStorage.getItem("token");

      if (storedUser) {
        setUser(JSON.parse(storedUser));
      } else if (token) {
        const fetchedUser = await fetchUserData(token);
        if (fetchedUser) {
          setUser(fetchedUser);
          localStorage.setItem("user", JSON.stringify(fetchedUser));
        }
      }
    };

    if (isAuthenticated) {
      loadUserData();
    }
  }, [isAuthenticated]);

  if (!isAuthenticated) {
    navigate("/login");
    return null;
  }

  return (
    <div className="flex">
      <Sidebar user={user} isCollapsed={isSidebarCollapsed} toggleSidebar={toggleSidebar} />
      <main
        className={`flex-1 p-6 min-h-screen bg-gray-50 transition-all duration-300 ${
          isSidebarCollapsed ? "ml-16" : "ml-64"
        }`}
      >
        <Outlet />
      </main>
      <AiAssistant />
    </div>
  );
};

export default AuthenticatedLayout;