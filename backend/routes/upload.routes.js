const express = require('express');
const multer = require('multer');
const { PutObjectCommand } = require('@aws-sdk/client-s3');
const { s3Client, bucketName } = require('../config/s3Config');

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

router.post('/single', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    if (!bucketName) {
      return res.status(500).json({ error: 'S3 bucket name not configured' });
    }

    const fileKey = `${Date.now()}-${req.file.originalname}`;
    const params = {
      Bucket: bucketName,
      Key: fileKey,
      Body: req.file.buffer,
      ContentType: req.file.mimetype,
    };

    await s3Client.send(new PutObjectCommand(params));
    
    // Generate the S3 URL
    const fileUrl = `https://${bucketName}.s3.amazonaws.com/${fileKey}`;
    
    res.json({
      message: 'File uploaded successfully',
      filename: fileKey,
      url: fileUrl
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Error uploading file: ' + error.message });
  }
});

module.exports = router;
