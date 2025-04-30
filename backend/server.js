const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const path = require("path");
require("dotenv").config({
    path: path.resolve(__dirname, "../.env"),
});

const rootRouter = require("./routes/index");
const networkingRoutes = require("./routes/networking.routes");
const uploadRouter = require("./routes/upload.routes");
const newsRoutes = require("./routes/news.routes");

if (!process.env.MONGODB_URI || !process.env.JWT_SECRET) {
    console.error("❌ Missing env vars");
    process.exit(1);
}

mongoose
    .connect(process.env.MONGODB_URI)
    .then(() => console.log("✅ MongoDB Connected"))
    .catch((err) => console.error("❌ MongoDB Error:", err));

const app = express();
app.use(
    cors()
);
app.use(express.json());
app.use("/api/v1", rootRouter);
app.use("/api/network", networkingRoutes);
app.use("/api/news", newsRoutes);
app.use("/api/upload", uploadRouter);
// Serve static files if frontend exists
app.use(express.static(path.join(__dirname, "../frontend/dist")));
app.get("*", (req, res) =>
    res.sendFile(path.join(__dirname, "../frontend/dist", "index.html"))
);

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(err.status || 500).json({
        error: process.env.NODE_ENV === 'development' ? err.message : 'Internal server error'
    });
});

const PORT = process.env.PORT || 3000;
// Listen on all network interfaces (0.0.0.0) instead of just localhost
app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on port ${PORT}`);
});

module.exports = app;
