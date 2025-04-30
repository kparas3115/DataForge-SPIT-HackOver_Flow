import { Button } from "@/components/ui/button";
import heroSectionBg from "../../assets/hero-section-bg.jpg"; // Adjust path as needed
import { motion } from "framer-motion";
import { Zap } from "lucide-react"; // Adding an icon for the badge
import { useNavigate } from "react-router-dom";

export function HeroSection() {
  const navigate = useNavigate();

  return (
    <section
      className="relative w-full py-12 md:py-24 lg:py-32 xl:py-48 bg-cover bg-center overflow-hidden"
      style={{ backgroundImage: `url(${heroSectionBg})` }}
    >
      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/60 to-transparent z-0"></div>

      <div className="container px-4 md:px-6 mx-auto relative z-10">
        <div className="flex flex-col items-center space-y-8 text-center text-white">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="space-y-4"
          >
            <h1 className="p-4 text-4xl font-extrabold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl/none bg-gradient-to-r from-white to-gray-200 bg-clip-text text-transparent">
              Welcome to Research<span className="font-hindi">साथी</span>
            </h1>
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="bg-white/60 text-blue-900 inline-block rounded-lg bg-blue-100/20 px-4 py-2 text-sm font-semibold tracking-wider text-blue-300 uppercase"
            >
              <Zap className="inline-block h-4 w-4 mr-2" />
              Research<span className="font-hindi">साथी</span>
            </motion.div>
            <p className="mx-auto max-w-lg text-lg leading-relaxed text-gray-100 md:text-xl">
              Unite fragmented research with a centralized platform for discovering articles, datasets, and experts.
              Enhance collaboration with AI-driven insights and real-time tools.
            </p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.8 }}
            className="space-x-6"
          >
            <Button
              onClick={() => navigate("/signup")}
              size="lg"
              className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700 rounded-lg shadow-md transition-all duration-300"
            >
              Get Started
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="px-8 py-3 bg-transparent text-white border-white hover:bg-white hover:text-blue-600 rounded-lg shadow-md transition-all duration-300"
            >
              Learn More
            </Button>
          </motion.div>
        </div>
      </div>
    </section>
  );
}