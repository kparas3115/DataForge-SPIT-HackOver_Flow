import React from "react";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { MessageCircle, ExternalLink } from "lucide-react";

const ResearcherMatches = () => {
  const matches = [
    {
      id: 1,
      name: "Dr. Sarah Parker",
      title: "AI Researcher",
      institution: "Stanford University",
      matchScore: 95,
      commonInterests: ["Machine Learning", "Neural Networks", "Deep Learning"],
      image: "https://api.dicebear.com/7.x/initials/svg?seed=Sarah Parker"
    },
    {
      id: 2,
      name: "Prof. James Wilson",
      title: "Data Science Lead",
      institution: "MIT",
      matchScore: 88,
      commonInterests: ["Big Data", "Data Mining", "Statistical Analysis"],
      image: "https://api.dicebear.com/7.x/initials/svg?seed=James Wilson"
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Research Matches</h1>
          <p className="text-gray-600">Discover researchers with similar interests and expertise</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {matches.map((match) => (
            <div key={match.id} className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center">
                  <img
                    src={match.image}
                    alt={match.name}
                    className="w-16 h-16 rounded-lg"
                  />
                  <div className="ml-4">
                    <h3 className="font-semibold text-lg">{match.name}</h3>
                    <p className="text-gray-600 text-sm">{match.title}</p>
                    <p className="text-gray-500 text-sm">{match.institution}</p>
                  </div>
                </div>
                <Badge className="bg-green-100 text-green-800 border-green-200">
                  {match.matchScore}% Match
                </Badge>
              </div>

              <div className="mb-4">
                <p className="text-sm text-gray-600 mb-2">Common Research Interests:</p>
                <div className="flex flex-wrap gap-2">
                  {match.commonInterests.map((interest, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {interest}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="flex gap-3">
                <Button variant="outline" className="flex-1">
                  <MessageCircle className="w-4 h-4 mr-2" />
                  Message
                </Button>
                <Button variant="outline" className="flex-1">
                  <ExternalLink className="w-4 h-4 mr-2" />
                  View Profile
                </Button>
                <Button className="flex-1">
                  Connect
                </Button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ResearcherMatches;
