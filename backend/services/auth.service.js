// services/auth.service.js
const bcrypt = require("bcryptjs");
const { User } = require("../schemas/users.schema");
const { generateToken } = require("../config");

/**
 * User signup service
 * @param {Object} userData - User data for registration
 * @returns {Object} - Result with user and token or error
 */
const signup = async (userData) => {
    const {
        username,
        firstName,
        lastName,
        password,
        shortBio,
        interests,
        instituteName,
        papers,
    } = userData;

    // Check if user exists
    const existingUser = await User.findOne({ username });
    if (existingUser) {
        return {
            success: false,
            status: 409,
            message: "Email already taken",
        };
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create new user
    const newUser = new User({
        username,
        firstName,
        lastName,
        password: hashedPassword,
        shortBio,
        interests,
        instituteName,
        papers,
    });

    await newUser.save();

    // Generate token
    const token = generateToken(newUser);

    return {
        success: true,
        status: 201,
        message: "User created successfully",
        token,
    };
};

/**
 * User signin service
 * @param {string} username - User email
 * @param {string} password - User password
 * @returns {Object} - Result with token or error
 */
const signin = async (username, password) => {
    // Find user by username/email
    const user = await User.findOne({ username });
    if (!user) {
        return {
            success: false,
            status: 401,
            message: "Invalid email or password",
        };
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
        return {
            success: false,
            status: 401,
            message: "Invalid email or password",
        };
    }

    // Generate token
    const token = generateToken(user);

    return {
        success: true,
        status: 200,
        message: "User logged in successfully!",
        token,
    };
};

/**
 * Get user profile service
 * @param {string} userId - User ID
 * @returns {Object} - Result with user data or error
 */
const getUserProfile = async (userId) => {
    const user = await User.findById(userId).select("-password");
    if (!user) {
        return {
            success: false,
            status: 404,
            message: "User not found",
        };
    }

    return {
        success: true,
        status: 200,
        user,
    };
};

/**
 * Update user profile service
 * @param {string} userId - User ID
 * @param {Object} updateData - Data to update
 * @returns {Object} - Result with updated user or error
 */
const updateUserProfile = async (userId, updateData) => {
    // Check if there's anything to update
    if (Object.keys(updateData).length === 0) {
        return {
            success: false,
            status: 400,
            message: "No valid fields to update",
        };
    }

    const updatedUser = await User.findByIdAndUpdate(userId, updateData, {
        new: true,
        runValidators: true,
    }).select("-password");

    if (!updatedUser) {
        return {
            success: false,
            status: 404,
            message: "User not found",
        };
    }

    return {
        success: true,
        status: 200,
        message: "User updated successfully",
        user: updatedUser,
    };
};

module.exports = {
    signup,
    signin,
    getUserProfile,
    updateUserProfile,
};
