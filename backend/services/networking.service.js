// services/networking.service.js
const { User } = require("../schemas/users.schema");

/**
 * Get all users with pagination, sorting, and filtering
 * @param {Object} options - Pagination and sorting options
 * @param {number} options.limit - Maximum number of users to return
 * @param {number} options.skip - Number of users to skip
 * @param {string} options.sortBy - Field to sort by
 * @param {number} options.sortOrder - Sort order (1 for ascending, -1 for descending)
 * @returns {Promise<Object>} - Result object with users and metadata
 */
const getAllUsers = async (options = {}) => {
    try {
        const {
            limit = 10,
            skip = 0,
            sortBy = "firstName",
            sortOrder = 1,
        } = options;

        // Create sort object
        const sort = {};
        sort[sortBy] = sortOrder;

        // Find users with pagination and sorting
        const users = await User.find({}, "-password")
            .sort(sort)
            .skip(skip)
            .limit(limit);

        // Get total count for pagination metadata
        const total = await User.countDocuments();

        return {
            success: true,
            status: 200,
            users,
            meta: {
                total,
                limit,
                skip,
                hasMore: skip + users.length < total,
            },
        };
    } catch (error) {
        console.error("Get All Users Service Error:", error);
        return {
            success: false,
            status: 500,
            message: "Failed to fetch users",
        };
    }
};

/**
 * Search users by query string, institute name, or interests
 * @param {Object} searchParams - Search parameters
 * @param {string} searchParams.query - General search query
 * @param {string} searchParams.instituteName - Institute name to filter by
 * @param {string} searchParams.interests - Interests to filter by
 * @param {number} searchParams.limit - Maximum number of users to return
 * @param {number} searchParams.skip - Number of users to skip
 * @returns {Promise<Object>} - Result object with users and metadata
 */
const searchUsers = async (searchParams = {}) => {
    try {
        const {
            query,
            instituteName,
            interests,
            limit = 10,
            skip = 0,
        } = searchParams;

        // Build search filter
        const filter = {};

        if (query) {
            filter.$or = [
                { firstName: { $regex: query, $options: "i" } },
                { lastName: { $regex: query, $options: "i" } },
                { email: { $regex: query, $options: "i" } },
            ];
        }

        if (instituteName) {
            filter.instituteName = { $regex: instituteName, $options: "i" };
        }

        if (interests) {
            filter.interests = { $regex: interests, $options: "i" };
        }

        // Find users with filter, pagination
        const users = await User.find(filter, "-password")
            .skip(skip)
            .limit(limit);

        // Get total count for pagination metadata
        const total = await User.countDocuments(filter);

        return {
            success: true,
            status: 200,
            users,
            meta: {
                total,
                limit,
                skip,
                hasMore: skip + users.length < total,
            },
        };
    } catch (error) {
        console.error("Search Users Service Error:", error);
        return {
            success: false,
            status: 500,
            message: "Failed to search users",
        };
    }
};

/**
 * Get user by ID
 * @param {string} userId - User ID
 * @returns {Promise<Object>} - Result object with user data
 */
const getUserById = async (userId) => {
    try {
        const user = await User.findById(userId, "-password");

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
    } catch (error) {
        console.error("Get User By ID Service Error:", error);
        return {
            success: false,
            status: 500,
            message: "Failed to fetch user",
        };
    }
};

module.exports = {
    getAllUsers,
    searchUsers,
    getUserById,
};
