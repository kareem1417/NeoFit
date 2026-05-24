require('dotenv').config();
const { MongoClient } = require('mongodb');
const app = require('./app');

const port = process.env.PORT || 3000;
const uri = process.env.MONGO_URI;

async function startServer() {
  try {
    const client = new MongoClient(uri);
    await client.connect();
    console.log('Successfully connected to MongoDB!');
    
    app.listen(port, () => {
      console.log(`App running on port ${port}...`);
    });
  } catch (err) {
    console.error('Failed to connect to MongoDB', err);
    process.exit(1);
  }
}

startServer();
