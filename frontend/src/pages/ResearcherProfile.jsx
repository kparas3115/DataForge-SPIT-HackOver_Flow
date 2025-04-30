import React from "react";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { 
  User, Mail, Building, BookOpen, 
  Award, MessageCircle, Users, Link 
} from "lucide-react";

const ResearcherProfile = () => {
  const researcher = {
    name: "Dr. Sarah Connor",
    title: "AI Research Scientist",
    institution: "MIT",
    bio: "Focusing on machine learning applications in healthcare",
    interests: ["AI", "Healthcare", "Machine Learning", "Neural Networks"],
    papers: [
      {
        title: "Deep Learning in Medical Imaging",
        citations: 156,
        year: 2023,
      }
    ],
    stats: {
      publications: 45,
      citations: 1200,
      collaborators: 23,
    },
    badges: ["AI Expert", "Top Contributor", "Healthcare Pioneer"]
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Profile Header */}
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="bg-gradient-to-r from-blue-600 to-blue-800 h-32"></div>
          <div className="relative px-6 pb-6">
            <div className="flex justify-between items-end">
              <div className="flex items-end">
                <div className="relative -mt-16">
                  <img
                    src={`https://api.dicebear.com/7.x/initials/svg?seed=${researcher.name}`}
                    alt="Profile"
                    className="w-32 h-32 rounded-lg border-4 border-white bg-white"
                  />
                </div>
                <div className="ml-4 mb-4">
                  <h1 className="text-2xl font-bold">{researcher.name}</h1>
                  <p className="text-gray-600">{researcher.title}</p>
                </div>
              </div>
              <div className="flex space-x-3">
                <Button variant="outline" className="flex items-center">
                  <MessageCircle className="w-4 h-4 mr-2" />
                  Message
                </Button>
                <Button className="flex items-center">
                  <Users className="w-4 h-4 mr-2" />
                  Connect
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="mt-6 grid grid-cols-3 gap-6">
          {/* Left Column - Info */}
          <div className="col-span-2 space-y-6">
            {/* About */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">About</h2>
              <p className="text-gray-600">{researcher.bio}</p>
            </div>

            {/* Publications */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Recent Publications</h2>
              {researcher.papers.map((paper, index) => (
                <div key={index} className="border-b py-4 last:border-0">
                  <h3 className="font-medium">{paper.title}</h3>
                  <div className="flex items-center text-sm text-gray-500 mt-2">
                    <span>Citations: {paper.citations}</span>
                    <span className="mx-2">•</span>
                    <span>{paper.year}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column - Stats & Details */}
          <div className="space-y-6">
            {/* Quick Stats */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Stats</h2>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-gray-600">Publications</span>
                  <span className="font-medium">{researcher.stats.publications}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Citations</span>
                  <span className="font-medium">{researcher.stats.citations}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Collaborators</span>
                  <span className="font-medium">{researcher.stats.collaborators}</span>
                </div>
              </div>
            </div>

            {/* Interests */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Research Interests</h2>
              <div className="flex flex-wrap gap-2">
                {researcher.interests.map((interest, index) => (
                  <Badge key={index} variant="secondary" className="px-3 py-1">
                    {interest}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Badges */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Achievements</h2>
              <div className="space-y-3">
                {researcher.badges.map((badge, index) => (
                  <div key={index} className="flex items-center">
                    <Award className="w-5 h-5 text-yellow-500 mr-2" />
                    <span>{badge}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResearcherProfile;
