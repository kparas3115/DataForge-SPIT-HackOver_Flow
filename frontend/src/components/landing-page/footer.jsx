import { Github, Twitter, Linkedin } from "lucide-react";

export function Footer() {
  return (
    <footer className="w-full py-6 bg-blue-950/20 border-t">
      <div className="container px-4 md:px-6 mx-auto">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Research<span className="font-hindi">साथी</span></h3>
            <p className="text-sm text-gray-800">
              A centralized platform for research collaboration and discovery.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="text-gray-800 hover:text-blue-600">
                <Github className="h-5 w-5" />
                <span className="sr-only">GitHub</span>
              </a>
              <a href="#" className="text-gray-800 hover:text-blue-600">
                <Twitter className="h-5 w-5" />
                <span className="sr-only">Twitter</span>
              </a>
              <a href="#" className="text-gray-800 hover:text-blue-600">
                <Linkedin className="h-5 w-5" />
                <span className="sr-only">LinkedIn</span>
              </a>
            </div>
          </div>
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Resources</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Documentation
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Components
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Examples
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Templates
                </a>
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Company</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  About
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Blog
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Careers
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Contact
                </a>
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Legal</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Terms of Service
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-800 hover:text-blue-600">
                  Cookie Policy
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-8 border-t pt-8 flex flex-col sm:flex-row justify-between items-center">
          <p className="text-xs text-gray-800">
            © {new Date().getFullYear()} Research<span className="font-hindi">साथी</span>. All rights reserved.
          </p>
          <p className="text-xs text-gray-800 mt-4 sm:mt-0">
            Made with ❤️ for researchers
          </p>
        </div>
      </div>
    </footer>
  );
}