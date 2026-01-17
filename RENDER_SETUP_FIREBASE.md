# Setting up Firebase Credentials on Render

## Step 1: Get your Firebase Service Account JSON

The service account key is stored locally in:
```
traceability/serviceAccountKey.json
```

## Step 2: Add Environment Variable on Render

1. Go to your Render dashboard: https://dashboard.render.com/
2. Click on your backend service: **spice-purity-server**
3. Go to **Environment** tab
4. Click **Add Environment Variable**
5. Add the following:
   - **Key**: `FIREBASE_CREDENTIALS`
   - **Value**: Copy and paste the ENTIRE contents of `serviceAccountKey.json` as a single line JSON string

Example format:
```
{"type":"service_account","project_id":"spice-sense-69","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"...","client_id":"...","auth_uri":"...","token_uri":"...","auth_provider_x509_cert_url":"...","client_x509_cert_url":"..."}
```

6. Click **Save Changes**

## Step 3: Verify Deployment

After Render redeploys with the environment variable:

1. Check the logs: https://dashboard.render.com/web/[your-service-id]/logs
2. Look for: `✓ Firebase initialized successfully`
3. If you see `✗ Firebase initialization failed`, check the environment variable format

## Testing the API

Test authentication:
```bash
# This should return 401 Unauthorized (auth required)
curl https://spice-purity-server.onrender.com/api/batches

# Login from frontend and test with token
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" https://spice-purity-server.onrender.com/api/auth/me
```

## Database Location

SQLite database will be created at:
```
/opt/render/project/src/raw_sensor_model/server/traceability.db
```

Note: Render's free tier uses ephemeral storage, so the database resets on each deploy. For production, consider upgrading to a persistent disk or using PostgreSQL.
