// routes/news.routes.js
const express = require("express");
const router = express.Router();
const { authMiddleware } = require("../middleware");
const newsService = require("../services/news.service");
const zod = require("zod");

// Validation schema for news query parameters
const newsQuerySchema = zod.object({
    category: zod.string().optional(),
    limit: zod.string().regex(/^\d+$/).transform(Number).optional(),
});

/**
 * @route GET /api/news
 * @desc Get latest research breakthrough news
 * @access Private (requires authentication)
 */
router.get("/", authMiddleware, async (req, res) => {
    try {
        const validation = newsQuerySchema.safeParse(req.query);
        if (!validation.success) {
            return res.status(400).json({
                message: "Invalid query parameters",
                errors: validation.error.errors,
            });
        }

        const result = await newsService.getResearchNews(validation.data);
        return res
            .status(result.status)
            .json(
                result.success
                    ? { news: result.news, meta: result.meta }
                    : { message: result.message }
            );
    } catch (error) {
        console.error("Get Research News Error:", error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});

module.exports = router;
