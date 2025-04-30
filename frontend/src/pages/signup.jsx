// pages/SignupPage.jsx
import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Eye, EyeOff, Loader2, Trash2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const signupSchema = z
  .object({
    name: z.string().min(2, { message: "Name must be at least 2 characters" }),
    email: z.string().email({ message: "Please enter a valid email address" }),
    password: z.string().min(8, { message: "Password must be at least 6 characters" }),
    confirmPassword: z.string(),
    shortBio: z.string().min(1, { message: "Short bio is required" }).max(2000),
    interests: z.string().min(1, { message: "At least one interest is required" }).transform((val) => val.split(",").map((item) => item.trim())),
    instituteName: z.string().min(1, { message: "Institute name is required" }),
    papers: z.array(z.object({
      name: z.string().min(1, { message: "Paper name is required" }),
      url: z.string().url({ message: "Please enter a valid URL" }),
    })).min(1, { message: "At least one paper is required" }),
    connections: z.array(z.string()).optional(), // Optional field for connections (array of user IDs)
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"],
  });

export default function SignupPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [showAdditionalFields, setShowAdditionalFields] = useState(false);
  const [papers, setPapers] = useState([{ name: "", url: "" }]);
  const navigate = useNavigate();

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
    trigger,
  } = useForm({
    resolver: zodResolver(signupSchema),
    defaultValues: {
      name: "",
      email: "",
      password: "",
      confirmPassword: "",
      shortBio: "",
      interests: "",
      instituteName: "",
      papers: [{ name: "", url: "" }],
      connections: [], // Default to empty array
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

  const handleProceed = async () => {
    const isValid = await trigger(["name", "email", "password", "confirmPassword"]);
    if (isValid) {
      setShowAdditionalFields(true);
    }
  };

  const addPaper = () => {
    setPapers([...papers, { name: "", url: "" }]);
    setValue("papers", [...papers, { name: "", url: "" }]);
  };

  const handlePaperChange = (index, field, value) => {
    const updatedPapers = papers.map((paper, i) =>
      i === index ? { ...paper, [field]: value } : paper
    );
    setPapers(updatedPapers);
    setValue("papers", updatedPapers);
  };

  const deletePaper = (index) => {
    if (papers.length === 1) return;
    const updatedPapers = papers.filter((_, i) => i !== index);
    setPapers(updatedPapers);
    setValue("papers", updatedPapers);
  };

  const onSubmit = async (data) => {
    setIsLoading(true);
    try {
      const nameParts = data.name.trim().split(" ");
      const firstName = nameParts[0] || "";
      const lastName = nameParts.slice(1).join(" ") || "";

      const payload = {
        username: data.email,
        firstName,
        lastName,
        password: data.password,
        shortBio: data.shortBio,
        interests: data.interests.join(","),
        instituteName: data.instituteName,
        papers: data.papers,
        connections: data.connections || [], // Include connections (defaults to empty array)
      };

      const res = await axios.post("http://localhost:3000/api/v1/users/signup", payload);
      const token = res.data.token;
      localStorage.setItem("token", token);

      // Fetch user data after signup
      const user = await fetchUserData(token);
      if (user) {
        localStorage.setItem("user", JSON.stringify(user));
      }

      navigate("/");
    } catch (err) {
      alert(err.response?.data?.message || "Signup failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <Card className="w-full max-w-2xl shadow-lg rounded-lg border border-gray-200">
        <CardHeader className="bg-white border-b border-gray-200 p-6">
          <CardTitle className="text-2xl font-semibold text-gray-800 text-center">
            Create Your Research Profile
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {/* Core Fields */}
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name" className="text-sm font-medium text-gray-700">
                  Full Name
                </Label>
                <Input
                  id="name"
                  type="text"
                  placeholder="John Doe"
                  className="rounded-md border-gray-300 focus:ring-blue-500 focus:border-blue-500 transition duration-200"
                  {...register("name")}
                />
                {errors.name && <p className="text-sm text-red-500">{errors.name.message}</p>}
              </div>

              <div className="space-y-2">
                <Label htmlFor="email" className="text-sm font-medium text-gray-700">
                  Email
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="name@example.com"
                  className="rounded-md border-gray-300 focus:ring-blue-500 focus:border-blue-500 transition duration-200"
                  {...register("email")}
                />
                {errors.email && <p className="text-sm text-red-500">{errors.email.message}</p>}
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-sm font-medium text-gray-700">
                  Password
                </Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="••••••••"
                    className="rounded-md border-gray-300 focus:ring-blue-500 focus:border-blue-500 transition duration-200"
                    {...register("password")}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-gray-100"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </Button>
                </div>
                {errors.password && <p className="text-sm text-red-500">{errors.password.message}</p>}
              </div>

              <div className="space-y-2">
                <Label htmlFor="confirmPassword" className="text-sm font-medium text-gray-700">
                  Confirm Password
                </Label>
                <div className="relative">
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? "text" : "password"}
                    placeholder="••••••••"
                    className="rounded-md border-gray-300 focus:ring-blue-500 focus:border-blue-500 transition duration-200"
                    {...register("confirmPassword")}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-gray-100"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </Button>
                </div>
                {errors.confirmPassword && <p className="text-sm text-red-500">{errors.confirmPassword.message}</p>}
              </div>

              <Button
                type="button"
                onClick={handleProceed}
                className="w-full bg-black hover:bg-gray-600 text-white rounded-md py-2 transition duration-200"
              >
                Proceed
              </Button>
            </div>

            {/* Additional Fields */}
            <AnimatePresence>
              {showAdditionalFields && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.5 }}
                  className="space-y-4 border-t pt-6 border-gray-200"
                >
                  <div className="space-y-2">
                    <Label htmlFor="shortBio" className="text-sm font-medium text-gray-700">
                      Short Bio
                    </Label>
                    <Textarea
                      id="shortBio"
                      placeholder="Tell us about yourself (max 2000 chars)"
                      maxLength={2000}
                      className="rounded-md border-gray-300 focus:ring-blue-500 focus:border-blue-500 transition duration-200"
                      {...register("shortBio")}
                    />
                    <p className="text-sm text-gray-500">
                      {watch("shortBio")?.length || 0}/2000
                    </p>
                    {errors.shortBio && <p className="text-sm text-red-500">{errors.shortBio.message}</p>}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="interests" className="text-sm font-medium text-gray-700">
                      Interests (comma-separated)
                    </Label>
                    <Input
                      id="interests"
                      type="text"
                      placeholder="e.g., AI, Machine Learning, Blockchain"
                      className="rounded-md border-gray-300 focus:ring-blue-500 focus:border-blue-500 transition duration-200"
                      {...register("interests")}
                    />
                    {errors.interests && <p className="text-sm text-red-500">{errors.interests.message}</p>}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="instituteName" className="text-sm font-medium text-gray-700">
                      Institute Name
                    </Label>
                    <Input
                      id="instituteName"
                      type="text"
                      placeholder="Your institution"
                      className="rounded-md border-gray-300 focus:ring-blue-500 focus:border-blue-500 transition duration-200"
                      {...register("instituteName")}
                    />
                    {errors.instituteName && <p className="text-sm text-red-500">{errors.instituteName.message}</p>}
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm font-medium text-gray-700">Research Papers</Label>
                    {papers.map((paper, index) => (
                      <div key={index} className="flex gap-2 mb-2 items-center flex-col sm:flex-row">
                        <Input
                          placeholder="Paper Name"
                          value={paper.name}
                          onChange={(e) => handlePaperChange(index, "name", e.target.value)}
                          className="rounded-md border-gray-300 focus:ring-blue-500 focus:border-blue-500 transition duration-200 flex-1"
                        />
                        <Input
                          placeholder="Paper URL"
                          value={paper.url}
                          onChange={(e) => handlePaperChange(index, "url", e.target.value)}
                          className="rounded-md border-gray-300 focus:ring-blue-500 focus:border-blue-500 transition duration-200 flex-1"
                        />
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="text-red-500 hover:text-red-700 hover:bg-red-100"
                          onClick={() => deletePaper(index)}
                          disabled={papers.length === 1}
                        >
                          <Trash2 size={16} />
                        </Button>
                      </div>
                    ))}
                    {errors.papers && <p className="text-sm text-red-500">{errors.papers.message}</p>}
                    <Button
                      type="button"
                      onClick={addPaper}
                      className="bg-blue-500 hover:bg-blue-600 text-white rounded-md px-4 py-2 transition duration-200"
                    >
                      Add Another Paper
                    </Button>
                  </div>

                  {/* No UI for connections during signup as it’s initialized empty */}
                  <Button
                    type="submit"
                    className="w-full bg-blue-500 hover:bg-blue-600 text-white rounded-md py-2 transition duration-200"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Creating account...
                      </>
                    ) : (
                      "Create Account"
                    )}
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}