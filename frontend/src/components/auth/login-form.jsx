import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import axios from "axios";
import { Eye, EyeOff, Loader2, ArrowRightLeft, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { useNavigate } from "react-router-dom";

const loginSchema = z.object({
  email: z.string().email({ message: "Please enter a valid email address" }),
  password: z.string().min(8, { message: "Password must be at least 8 characters" }),
  rememberMe: z.boolean().optional(),
});

export function LoginForm() {
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: "",
      password: "",
      rememberMe: false,
    },
  });

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

  const onSubmit = async (data) => {
    setIsLoading(true);
    try {
      const res = await axios.post("http://localhost:3000/api/v1/users/signin", {
        username: data.email,
        password: data.password,
      });
      const token = res.data.token;
      localStorage.setItem("token", token);

      // Fetch user data after login and store it
      const user = await fetchUserData(token);
      if (user) {
        localStorage.setItem("user", JSON.stringify(user));
      } else {
        console.warn("Failed to fetch user data after login");
      }

      navigate("/research-papers");
    } catch (err) {
      alert(err.response?.data?.message || "Sign-in failed");
    } finally {
      setIsLoading(false);
    }
  };
  const navigate = useNavigate();

  const handleGoBack = () => {
    navigate("/"); // Redirects to the root path (localhost:5173/)
  };
  return (
    <Card className="w-full max-w-md mx-auto p-2">
<div
      className="inline-flex items-center gap-2 p-4 cursor-pointer hover:bg-gray-100 rounded-lg transition-colors duration-200"
      onClick={handleGoBack}
    >
      <ArrowLeft className="w-5 h-5 text-gray-600" />
      <span className="text-sm font-medium text-gray-700">Go Back</span>
    </div>
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl font-bold text-center">Login</CardTitle>
        <CardDescription className="text-center">Enter your email and password to access your account</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input id="email" type="email" placeholder="name@example.com" {...register("email")} />
            {errors.email && <p className="text-sm text-red-500">{errors.email.message}</p>}
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="password">Password</Label>
              <Button variant="link" className="p-0 h-auto text-xs" type="button">
                Forgot password?
              </Button>
            </div>
            <div className="relative">
              <Input
                id="password"
                type={showPassword ? "text" : "password"}
                placeholder="••••••••"
                {...register("password")}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                <span className="sr-only">{showPassword ? "Hide password" : "Show password"}</span>
              </Button>
            </div>
            {errors.password && <p className="text-sm text-red-500">{errors.password.message}</p>}
          </div>
          <div className="flex items-center space-x-2">
            <Checkbox id="rememberMe" {...register("rememberMe")} />
            <Label htmlFor="rememberMe" className="text-sm font-normal">
              Remember me for 30 days
            </Label>
          </div>
          <Button type="submit" className="w-full bg-black text-white hover:bg-black/40" disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Logging in...
              </>
            ) : (
              "Login"
            )}
          </Button>
        </form>
      </CardContent>
      <CardFooter className="flex flex-col">
        <p className="mt-4 text-center text-sm text-gray-600">
          Don't have an account?{" "}
          <a href="/signup" className="font-medium text-primary hover:underline">
            Sign up
          </a>
        </p>
      </CardFooter>
    </Card>
  );
}