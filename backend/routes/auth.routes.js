// routes/auth.routes.js
const express = require("express");
const router = express.Router();
const { authMiddleware } = require("../middleware");
const authService = require("../services/auth.service");
const { userValidation } = require("../schemas/validation.schema");

// Routes
router.post("/signup", async (req, res) => {
    try {
        const validation = userValidation.signup.safeParse(req.body);
        if (!validation.success) {
            return res.status(400).json({
                message: "Invalid input data",
                errors: validation.error.errors,
            });
        }

        const result = await authService.signup(req.body);
        return res.status(result.status).json({
            message: result.message,
            token: result.token,
        });
    } catch (error) {
        console.error("Signup Error:", error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});

router.post("/signin", async (req, res) => {
    try {
        const validation = userValidation.signin.safeParse(req.body);
        if (!validation.success) {
            return res.status(400).json({
                message: "Invalid credentials",
                errors: validation.error.errors,
            });
        }

        const result = await authService.signin(
            req.body.username,
            req.body.password
        );
        return res.status(result.status).json({
            message: result.message,
            token: result.success ? result.token : undefined,
        });
    } catch (error) {
        console.error("Signin Error:", error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});

router.get("/me", authMiddleware, async (req, res) => {
    try {
        const result = await authService.getUserProfile(req.userId);
        return res
            .status(result.status)
            .json(
                result.success
                    ? { user: result.user }
                    : { message: result.message }
            );
    } catch (error) {
        console.error("Get User Error:", error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});

router.put("/update", authMiddleware, async (req, res) => {
    try {
        const validation = userValidation.updateProfile.safeParse(req.body);
        if (!validation.success) {
            return res.status(400).json({
                message: "Invalid input data",
                errors: validation.error.errors,
            });
        }

        const result = await authService.updateUserProfile(
            req.userId,
            req.body
        );
        return res.status(result.status).json({
            message: result.message,
            user: result.success ? result.user : undefined,
        });
    } catch (error) {
        console.error("Update User Error:", error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});

module.exports = router;
