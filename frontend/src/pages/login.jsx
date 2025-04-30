import { LoginForm } from "@/components/auth/login-form";
import { Footer } from "@/components/landing-page/footer";

export default function LoginPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <main className="flex-1 py-12">
        <div className="container mx-auto px-4 md:px-6">
          <div className="flex justify-center">
            <LoginForm />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}