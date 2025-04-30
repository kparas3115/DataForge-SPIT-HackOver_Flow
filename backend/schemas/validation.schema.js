// schemas/validation.schema.js
const zod = require("zod");

const userValidation = {
    // User signup validation schema
    signup: zod.object({
        username: zod.string().email({ message: "Invalid email format" }),
        firstName: zod.string().min(1, { message: "First name is required" }),
        lastName: zod.string().min(1, { message: "Last name is required" }),
        password: zod
            .string()
            .min(6, { message: "Password must be at least 6 characters" }),
        shortBio: zod
            .string()
            .max(2000, { message: "Bio must be less than 2000 characters" }),
        interests: zod.string(),
        instituteName: zod.string(),
        papers: zod.array(
            zod.object({
                name: zod
                    .string()
                    .min(1, { message: "Paper name is required" }),
                url: zod.string().url({ message: "Invalid URL format" }),
            })
        ),
    }),

    // User signin validation schema
    signin: zod.object({
        username: zod.string().email({ message: "Invalid email format" }),
        password: zod
            .string()
            .min(6, { message: "Password must be at least 6 characters" }),
    }),

    // Update user profile validation schema
    updateProfile: zod.object({
        shortBio: zod
            .string()
            .max(2000, { message: "Bio must be less than 2000 characters" })
            .optional(),
        interests: zod.string().optional(),
        instituteName: zod.string().optional(),
        papers: zod
            .array(
                zod.object({
                    name: zod
                        .string()
                        .min(1, { message: "Paper name is required" }),
                    url: zod.string().url({ message: "Invalid URL format" }),
                })
            )
            .optional(),
    }),
};

module.exports = {
    userValidation,
};
