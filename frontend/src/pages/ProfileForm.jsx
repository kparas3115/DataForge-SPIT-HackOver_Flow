import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import axios from "axios";
import { useNavigate } from "react-router-dom";

// Schema validation
const updateProfileSchema = z.object({
    shortBio: z.string().min(1, "Short bio is required").max(2000),
    interests: z
        .string()
        .min(1, "At least one interest is required")
        .transform((val) => val.split(",").map((item) => item.trim())),
    instituteName: z.string().min(1, "Institute name is required"),
    papers: z
        .array(
            z.object({
                name: z.string().min(1, "Paper name is required"),
                url: z.string().url("Please enter a valid URL"),
            })
        )
        .min(1, "At least one paper is required"),
});

// Fetch user data
const fetchUserData = async (token) => {
    try {
        const res = await axios.get("http://localhost:3000/api/v1/users/me", {
            headers: { Authorization: `Bearer ${token}` },
        });
        return res.data.user;
    } catch (err) {
        console.error(
            "Error fetching user data:",
            err.response?.data || err.message
        );
        return null;
    }
};

export default function UpdateProfile() {
    const [isLoading, setIsLoading] = useState(false);
    const [papers, setPapers] = useState([{ name: "", url: "" }]);
    const navigate = useNavigate();
    const token = localStorage.getItem("token");

    const {
        register,
        handleSubmit,
        formState: { errors },
        setValue,
        watch,
    } = useForm({
        resolver: zodResolver(updateProfileSchema),
    });

    // Fetch and populate user data
    useEffect(() => {
        const loadUserData = async () => {
            if (!token) {
                navigate("/login");
                return;
            }
            const userData = await fetchUserData(token);
            if (userData) {
                setValue("shortBio", userData.shortBio || "");
                setValue(
                    "interests",
                    Array.isArray(userData.interests)
                        ? userData.interests.join(", ")
                        : userData.interests || ""
                );
                setValue("instituteName", userData.instituteName || "");
                const formattedPapers = userData.papers?.length
                    ? userData.papers
                    : [{ name: "", url: "" }];
                setPapers(formattedPapers);
                setValue("papers", formattedPapers);
            }
        };
        loadUserData();
    }, [token, navigate, setValue]);

    // Handle adding a new paper field
    const addPaper = () => {
        const newPapers = [...papers, { name: "", url: "" }];
        setPapers(newPapers);
        setValue("papers", newPapers);
    };

    // Handle paper input changes
    const handlePaperChange = (index, field, value) => {
        const updatedPapers = papers.map((paper, i) =>
            i === index ? { ...paper, [field]: value } : paper
        );
        setPapers(updatedPapers);
        setValue("papers", updatedPapers);
    };

    // Handle deleting a paper
    const deletePaper = (index) => {
        if (papers.length === 1) return;
        const updatedPapers = papers.filter((_, i) => i !== index);
        setPapers(updatedPapers);
        setValue("papers", updatedPapers);
    };

    // Handle form submission
    const onSubmit = async (data) => {
        setIsLoading(true);
        try {
            const userData = await fetchUserData(token);
            await axios.put(
                "http://localhost:3000/api/v1/users/update",
                {
                    userId: userData?.id || "",
                    shortBio: data.shortBio,
                    interests: Array.isArray(data.interests)
                        ? data.interests.join(", ")
                        : data.interests, // Convert back to string
                    instituteName: data.instituteName,
                    papers: data.papers.map((paper) => ({
                        name: paper.name,
                        url: paper.url,
                        ...(paper._id && { _id: paper._id }),
                    })),
                },
                {
                    headers: { Authorization: `Bearer ${token}` },
                }
            );

            alert("Profile updated successfully");
            navigate("/user-profile");
        } catch (err) {
            console.error("Update error:", err.response?.data || err.message);
            alert("Update failed. Please check the input data.");
        } finally {
            setIsLoading(false);
        }
    };
    const fetchUserData = async (token) => {
        try {
            const res = await axios.get(
                "http://localhost:3000/api/v1/users/me",
                {
                    headers: { Authorization: `Bearer ${token}` },
                }
            );
            return res.data.user;
        } catch (err) {
            console.error(
                "Error fetching user data:",
                err.response?.data || err.message
            );
            return null;
        }
    };
    return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center py-12 px-4">
            <Card className="w-full max-w-2xl shadow-xl rounded-lg bg-white border border-gray-300">
                <CardHeader className="p-6 bg-white border-b border-gray-200">
                    <CardTitle className="text-3xl font-semibold text-gray-800 text-center">
                        Update Profile
                    </CardTitle>
                </CardHeader>
                <CardContent className="p-6 space-y-6">
                    <form
                        onSubmit={handleSubmit(onSubmit)}
                        className="space-y-6"
                    >
                        <div className="space-y-4">
                            <div>
                                <Label className="text-gray-700">
                                    Short Bio
                                </Label>
                                <Textarea
                                    placeholder="Tell us about yourself..."
                                    {...register("shortBio")}
                                    className="border-gray-300 focus:ring-blue-500 focus:border-blue-500"
                                />
                                <p className="text-sm text-gray-500">
                                    {watch("shortBio")?.length || 0}/2000
                                </p>
                                {errors.shortBio && (
                                    <p className="text-sm text-red-500">
                                        {errors.shortBio.message}
                                    </p>
                                )}
                            </div>

                            <div>
                                <Label className="text-gray-700">
                                    Interests (comma-separated)
                                </Label>
                                <Input
                                    type="text"
                                    placeholder="e.g., AI, Machine Learning"
                                    {...register("interests")}
                                    className="border-gray-300 focus:ring-blue-500 focus:border-blue-500"
                                />
                                {errors.interests && (
                                    <p className="text-sm text-red-500">
                                        {errors.interests.message}
                                    </p>
                                )}
                            </div>

                            <div>
                                <Label className="text-gray-700">
                                    Institute Name
                                </Label>
                                <Input
                                    type="text"
                                    placeholder="Your institution"
                                    {...register("instituteName")}
                                    className="border-gray-300 focus:ring-blue-500 focus:border-blue-500"
                                />
                                {errors.instituteName && (
                                    <p className="text-sm text-red-500">
                                        {errors.instituteName.message}
                                    </p>
                                )}
                            </div>

                            <div>
                                <Label className="text-gray-700">
                                    Research Papers
                                </Label>
                                {papers.map((paper, index) => (
                                    <div
                                        key={index}
                                        className="flex gap-2 items-center"
                                    >
                                        <Input
                                            placeholder="Paper Name"
                                            value={paper.name}
                                            onChange={(e) =>
                                                handlePaperChange(
                                                    index,
                                                    "name",
                                                    e.target.value
                                                )
                                            }
                                            className="border-gray-300"
                                        />
                                        <Input
                                            placeholder="Paper URL"
                                            value={paper.url}
                                            onChange={(e) =>
                                                handlePaperChange(
                                                    index,
                                                    "url",
                                                    e.target.value
                                                )
                                            }
                                            className="border-gray-300"
                                        />
                                        <Button
                                            type="button"
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => deletePaper(index)}
                                        >
                                            <Trash2 size={16} />
                                        </Button>
                                    </div>
                                ))}
                                <Button
                                    type="button"
                                    onClick={addPaper}
                                    className="mt-2 bg-blue-500 text-white hover:bg-blue-600"
                                >
                                    Add Paper
                                </Button>
                            </div>

                            <Button
                                type="submit"
                                className="w-full bg-blue-500 text-white hover:bg-blue-600"
                                disabled={isLoading}
                            >
                                {isLoading ? "Updating..." : "Update Profile"}
                            </Button>
                        </div>
                    </form>
                </CardContent>
            </Card>
        </div>
    );
}
