import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { HeroSection } from "@/components/landing-page/hero-section";
import { FeatureSection } from "@/components/landing-page/feature-section";
import { TestimonialSection } from "@/components/landing-page/testimonial-section";
import { CTASection } from "@/components/landing-page/cta-section";
import { Footer } from "@/components/landing-page/footer";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { Bell, ChevronDown, User, Edit, Settings, LogOut } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from "react-router-dom"; // Add this import
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

export default function LandingPage() {
  const navigate = useNavigate(); // Add this
  const [scrolled, setScrolled] = useState(false);
  const { isAuthenticated, user } = useAuth();

  // Check token and auth status on mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    console.log('Current token:', token, 'isAuthenticated:', isAuthenticated);
  }, [isAuthenticated]);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navItems = [
    { label: "Features", id: "features" },
    { label: "Testimonials", id: "testimonials" },
    { label: "Contact Us", id: "contact" }
  ];

  // Smooth scroll function
  const scrollToSection = (id) => {
    const section = document.getElementById(id);
    if (section) {
      window.scrollTo({
        top: section.offsetTop - 80, // Adjusted for fixed navbar height
        behavior: "smooth"
      });
    }
  };

  // Separate function for navigation logic
  const handleNavigation = (e) => {
    e.preventDefault(); // Prevent any default behavior
    const token = localStorage.getItem('token');
    console.log('Navigation clicked, token:', token);
    
    if (token) {
      console.log('Has token - navigating to dashboard');
      navigate('/connections', { replace: true });
    } else {
      console.log('No token - navigating to signup');
      navigate('/signup', { replace: true });
    }
  };

  const handleGetStarted = async () => {
    const token = localStorage.getItem('token');
    console.log('Get Started clicked with token:', token);

    try {
      // Try to navigate even before the check, if token exists
      if (token) {
        console.log('Has token - attempting navigation to dashboard/connections');
        navigate('/connections');
        return;
      }
      console.log('No token - navigating to signup');
      navigate('/signup');
    } catch (error) {
      console.error('Navigation error:', error);
      // Fallback to signup if anything fails
      navigate('/signup');
    }
  };

  return (
      <div className="flex min-h-screen flex-col bg-gray-50">
          <header className="fixed w-full z-50 bg-white/60">
              <div className="container mx-auto px-4 h-20 flex items-center justify-between">
                  <motion.div whileHover={{ scale: 1.05 }}>
                      <Link to="/" className="flex items-center space-x-3">
                          <div className="relative">
                              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl blur-sm opacity-80" />
                              <div className="relative h-12 w-12 bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl flex items-center justify-center">
                                  <span className="text-2xl font-bold text-white">
                                      R
                                  </span>
                              </div>
                          </div>
                          <span className="text-xl font-bold text-gray-700">
                              Research<span className="font-hindi">साथी</span>
                          </span>
                      </Link>
                  </motion.div>

                  <nav className="hidden md:flex items-center space-x-8">
                      {navItems.map((item) => (
                          <motion.div
                              key={item.id}
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                          >
                              <button
                                  onClick={() => scrollToSection(item.id)}
                                  className="text-sm font-medium text-gray-800/90 hover:text-gray-500 transition-colors duration-300"
                              >
                                  {item.label}
                              </button>
                          </motion.div>
                      ))}
                  </nav>

                  <div className="flex items-center gap-4">
                      {isAuthenticated ? (
                          <div className="flex items-center gap-4">
                              <Button
                                  variant="ghost"
                                  size="icon"
                                  className="relative text-blue-800 hover:bg-white/10"
                              >
                                  <Bell className="h-5 w-5 text-blue-900" />
                                  <span className="absolute -top-1 -right-1 h-5 w-5 bg-blue-500 text-white text-xs rounded-full flex items-center justify-center">
                                      3
                                  </span>
                              </Button>

                              <DropdownMenu>
                                  <DropdownMenuTrigger asChild>
                                      <Button
                                          variant="ghost"
                                          className="flex items-center gap-2 px-2 hover:bg-gray-100"
                                      >
                                          <Avatar className="h-8 w-8">
                                              <AvatarImage
                                                  src={
                                                      user?.profileImage ||
                                                      "/default-avatar.png"
                                                  }
                                              />
                                              <AvatarFallback className="bg-gradient-to-br from-blue-500 to-blue-600 text-gray-800">
                                                  {user?.firstName?.[0]}
                                                  {user?.lastName?.[0]}
                                              </AvatarFallback>
                                          </Avatar>
                                          <div className="flex items-center gap-2">
                                              <div className="text-sm text-left hidden md:block">
                                                  <p className="font-medium text-gray-800">
                                                      {user?.firstName}{" "}
                                                      {user?.lastName}
                                                  </p>
                                                  <p className="text-xs text-gray-800/75">
                                                      {user?.instituteName ||
                                                          "Researcher"}
                                                  </p>
                                              </div>
                                              <ChevronDown className="h-4 w-4 text-gray-800/70" />
                                          </div>
                                      </Button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent
                                      align="end"
                                      className="w-72"
                                  >
                                      <div className="flex items-center gap-4 p-4">
                                          <Avatar className="h-16 w-16">
                                              <AvatarImage
                                                  src={
                                                      user?.profileImage ||
                                                      "/default-avatar.png"
                                                  }
                                              />
                                              <AvatarFallback className="bg-gradient-to-br from-blue-500 to-blue-600 text-white text-xl">
                                                  {user?.firstName?.[0]}
                                                  {user?.lastName?.[0]}
                                              </AvatarFallback>
                                          </Avatar>
                                          <div>
                                              <p className="font-semibold text-gray-900">
                                                  {user?.firstName}{" "}
                                                  {user?.lastName}
                                              </p>
                                              <p className="text-sm text-gray-500">
                                                  {user?.email}
                                              </p>
                                              <p className="text-xs text-gray-500 mt-1">
                                                  {user?.instituteName}
                                              </p>
                                          </div>
                                      </div>
                                      <div className="p-2">
                                          <div className="grid grid-cols-2 gap-2 mb-2">
                                              <div className="bg-gray-50 p-3 rounded-lg text-center">
                                                  <div className="font-semibold text-gray-900">
                                                      {user?.papers?.length ||
                                                          0}
                                                  </div>
                                                  <div className="text-xs text-gray-500">
                                                      Papers
                                                  </div>
                                              </div>
                                              <div className="bg-gray-50 p-3 rounded-lg text-center">
                                                  <div className="font-semibold text-gray-900">
                                                      {user?.experience || 0}
                                                  </div>
                                                  <div className="text-xs text-gray-500">
                                                      Years Exp.
                                                  </div>
                                              </div>
                                          </div>
                                      </div>
                                      <DropdownMenuSeparator />
                                      <DropdownMenuItem
                                          onClick={() =>
                                              navigate("/dashboard/connections")
                                          }
                                          className="gap-2"
                                      >
                                          <User className="h-4 w-4 text-blue-500" />{" "}
                                          Dashboard
                                      </DropdownMenuItem>
                                      <DropdownMenuItem
                                          onClick={() =>
                                              navigate("/update-profile")
                                          }
                                          className="gap-2"
                                      >
                                          <Edit className="h-4 w-4 text-blue-500" />{" "}
                                          Update Profile
                                      </DropdownMenuItem>
                                      <DropdownMenuItem
                                          onClick={() => navigate("/settings")}
                                          className="gap-2"
                                      >
                                          <Settings className="h-4 w-4 text-blue-500" />{" "}
                                          Settings
                                      </DropdownMenuItem>
                                      <DropdownMenuSeparator />
                                      <DropdownMenuItem className="text-red-600 gap-2">
                                          <LogOut className="h-4 w-4" /> Logout
                                      </DropdownMenuItem>
                                  </DropdownMenuContent>
                              </DropdownMenu>
                          </div>
                      ) : (
                          <>
                              <motion.div
                                  whileHover={{ scale: 1.05 }}
                                  whileTap={{ scale: 0.95 }}
                              >
                                  <Button
                                      variant="outline"
                                      className="border border-gray-500 text-gray-800 hover:bg-gray-500 hover:text-gray-500"
                                      onClick={() => navigate("/login")}
                                  >
                                      Login
                                  </Button>
                              </motion.div>

                              <motion.div
                                  whileHover={{ scale: 1.05 }}
                                  whileTap={{ scale: 0.95 }}
                              >
                                  <Button
                                      onClick={handleGetStarted}
                                      className="bg-white border border-blue-500 text-blue-600 hover:bg-white/90"
                                  >
                                      Get Started
                                  </Button>
                              </motion.div>
                          </>
                      )}
                  </div>
              </div>
          </header>

          <main className="flex-1 pt-20">
              <HeroSection />
              <div className="relative" id="features">
                  <div className="absolute inset-0 bg-gradient-to-b from-white to-gray-50 -skew-y-3 -z-10 transform" />
                  <FeatureSection />
              </div>
              <div id="testimonials">
                  <TestimonialSection />
              </div>
              <div id="pricing">
                  <CTASection />
              </div>
              <div id="contact">
                  <Footer />
              </div>
          </main>
      </div>
  );
}