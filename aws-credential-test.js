// Simple script to verify AWS credentials
import { RekognitionClient, ListCollectionsCommand } from '@aws-sdk/client-rekognition';

// Log the current AWS environment variables (only the first few characters for security)
console.log('AWS Credentials Check:');
if (process.env.AWS_ACCESS_KEY_ID) {
  console.log(`AWS_ACCESS_KEY_ID: ${process.env.AWS_ACCESS_KEY_ID.substring(0, 3)}...`);
} else {
  console.log('AWS_ACCESS_KEY_ID: Not set');
}

if (process.env.AWS_SECRET_ACCESS_KEY) {
  console.log(`AWS_SECRET_ACCESS_KEY: ${process.env.AWS_SECRET_ACCESS_KEY.substring(0, 3)}...`);
} else {
  console.log('AWS_SECRET_ACCESS_KEY: Not set');
}

console.log(`AWS_REGION: ${process.env.AWS_REGION || 'Not set'}`);

// Create a Rekognition client
const rekognition = new RekognitionClient({ region: process.env.AWS_REGION || 'us-east-1' });

// Make a simple API call to verify credentials
async function testAwsCredentials() {
  try {
    console.log('Testing AWS credentials with ListCollections API call...');
    const command = new ListCollectionsCommand({});
    const response = await rekognition.send(command);
    console.log('AWS API call successful!');
    console.log('Response:', response);
    return true;
  } catch (error) {
    console.error('AWS API call failed:', error.message);
    if (error.$metadata) {
      console.error('Error metadata:', error.$metadata);
    }
    return false;
  }
}

// Run the test
testAwsCredentials().then(result => {
  console.log(`Credential test ${result ? 'PASSED' : 'FAILED'}`);
});