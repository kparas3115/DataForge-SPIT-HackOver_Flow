import { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext({
  isAuthenticated: false,
  user: null,
  loading: true,
  checkAuth: () => {},
});

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const checkAuth = async () => {
    console.log('Checking auth...');
    const token = localStorage.getItem('token');
    console.log('Token in checkAuth:', token);

    if (!token) {
      console.log('No token found - setting unauthenticated');
      setIsAuthenticated(false);
      setUser(null);
      setLoading(false);
      return false;
    }

    try {
      const response = await axios.get('http://localhost:3000/api/v1/users/me', {
        headers: { 
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      console.log('Auth check successful:', response.data);
      setUser(response.data.user);
      setIsAuthenticated(true);
      setLoading(false);
      return true;
    } catch (error) {
      console.error('Auth check failed:', error);
      localStorage.removeItem('token');
      setIsAuthenticated(false);
      setUser(null);
      setLoading(false);
      return false;
    }
  };

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      checkAuth();
    } else {
      setLoading(false);
    }
  }, []);

  const value = {
    isAuthenticated,
    user,
    loading,
    checkAuth,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => useContext(AuthContext);
