const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true,
        trim: true,
    },
    password: {
        type: String,
        required: true,
        minLength: 6,
        maxLength: 100,
    },
    firstName: {
        type: String,
        required: true,
        trim: true,
        maxLength: 30,
    },
    lastName: {
        type: String,
        required: true,
        trim: true,
        maxLength: 30,
    },
    shortBio: {
        type: String,
        required: true,
        trim: true,
        maxLength: 2000,
    },
    interests: {
        type: String,
        required: true,
        trim: true,
    },
    instituteName: {
        type: String,
        required: true,
        trim: true,
    },
    papers: [
        {
            name: {
                type: String,
                required: true,
                trim: true,
            },
            url: {
                type: String,
                required: true,
                trim: true,
            },
        },
    ],
    connections: [
        {
            type: mongoose.Schema.Types.ObjectId,
            ref: "User", // Reference to other users
        },
    ],
});

const User = mongoose.model("User", userSchema);

module.exports = { User };

