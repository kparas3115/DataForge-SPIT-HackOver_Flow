const express = require("express");
const router = express.Router();
const userRouter = require("./auth.routes");

router.use("/users", userRouter);
router.get("/", (req, res) => {
  res.send("Working!");
});

module.exports = router;
