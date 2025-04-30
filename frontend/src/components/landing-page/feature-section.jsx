import { Code, Database, Users, Brain } from "lucide-react";

export function FeatureSection() {
  const features = [
    {
      icon: <Brain className="h-10 w-10 text-blue-600" />,
      title: "AI-Driven Discovery",
      description: "Leverage AI recommendations to uncover relevant research articles and datasets tailored to your interests.",
    },
    {
      icon: <Database className="h-10 w-10 text-blue-600" />,
      title: "Centralized Resources",
      description: "Access a unified database of articles, datasets, and expert profiles, eliminating fragmentation.",
    },
    {
      icon: <Users className="h-10 w-10 text-blue-600" />,
      title: "Interdisciplinary Collaboration",
      description: "Connect with experts and collaborate in real-time with shared workspaces and discussion forums.",
    },
    {
      icon: <Code className="h-10 w-10 text-blue-600" />,
      title: "Scalable & Secure",
      description: "Built for scalability and data privacy, supporting a growing research community safely.",
    },
  ];

  return (
    <section className="w-full py-12 md:py-24 lg:py-32 bg-white text-black">
      <div className="container px-4 md:px-6 mx-auto">
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <div className="space-y-2">
            <div className="inline-block rounded-lg bg-gray-100 px-3 py-1 text-sm font-semibold tracking-wider text-blue-600">
              Features
            </div>
            <h2 className="text-3xl font-bold tracking-tighter md:text-4xl">Key Solutions</h2>
            <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
            Research<span className="font-hindi">साथी</span> addresses the challenges of fragmented research with intuitive tools for discovery, collaboration, and accessibility.
            </p>
          </div>
        </div>
        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-2 lg:gap-12 py-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="flex flex-col items-center space-y-4 rounded-lg border p-6 shadow-sm transition-all hover:shadow-md"
            >
              <div className="p-2 rounded-full bg-gray-100">{feature.icon}</div>
              <h3 className="text-xl font-bold">{feature.title}</h3>
              <p className="text-gray-500 text-center">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}