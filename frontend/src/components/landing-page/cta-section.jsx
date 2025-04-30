import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export function CTASection() {
  return (
    <section className="w-full py-12 md:py-24 lg:py-32 bg-white text-black">
      <div className="container px-4 md:px-6 mx-auto">
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <div className="space-y-2">
            <h2 className="text-3xl font-bold tracking-tighter md:text-4xl/tight">
              Ready to Transform Research Collaboration?
            </h2>
            <p className="mx-auto max-w-[600px] text-gray-600 md:text-xl/relaxed">
              Join researchers worldwide in breaking down barriers with Research<span className="font-hindi">साथी</span> centralized platform.
            </p>
          </div>
          <div className="mx-auto w-full max-w-sm space-y-2">
            <form className="flex flex-col sm:flex-row gap-2">
              <Input
                type="email"
                placeholder="Enter your email"
                className="max-w-lg flex-1 border-gray-300 focus:ring-blue-500 focus:border-blue-500"
              />
              <Button type="submit" className="bg-blue-600 text-white hover:bg-blue-700">
                Get Started
              </Button>
            </form>
            <p className="text-xs text-gray-500">
              By subscribing, you agree to our terms and privacy policy.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}