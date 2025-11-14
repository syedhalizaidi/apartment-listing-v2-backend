# Apartment Listing BotAPI Endpoints

## How to Run

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key
CHROMA_DB_DIR=./chromadb
```

### 4. Run the Server

```bash
python run.py
```

### 5. Access API

Server runs at:

```
http://localhost:5000
```

## Project Structure

```
apartment-listing/
  app/
    api.py
    services/
  run.py
  requirements.txt
  Dockerfile
```


- `POST /api/webhook` - Webhook endpoint for Twilio WhatsApp messages
- `GET /api/listings` - Get all listings
- `POST /api/listings` - Add a new listing
- `GET /api/listings/<id>` - Get a specific listing
- `PUT /api/listings/<id>` - Update a listing
- `DELETE /api/listings/<id>` - Delete a listing

## Project Structure

```
.
├── app/                    # Main application package
│   ├── __init__.py        # Application factory
│   ├── api.py             # API routes and resources
│   ├── config.py          # Configuration settings
│   ├── dev/               # Development resources and test data
│   └── services/          # Business logic and services
├── static/                # Static files
├── .env                  # Environment variables
├── .gitignore            # Git ignore file
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
└── run.py               # Application entry point
```

## Environment Setup

1. **Database Setup**:

   - Ensure PostgreSQL is running
   - Create a new database for the application
   - Update the `DATABASE_URL` in `.env` with your database credentials
2. **Twilio Setup**:

   - Create a Twilio account
   - Set up WhatsApp Sandbox
   - Configure the webhook URL to point to your `/api/webhook` endpoint
3. **OpenAI Setup**:

   - Obtain an API key from OpenAI
   - Add it to your `.env` file

## Development

1. **Running the application locally**:

   ```bash
   python run.py
   ```
2. **Running tests**:

   ```bash
   # Run unit tests
   python -m pytest tests/
   ```
3. **Code style**:

   - Follow PEP 8 guidelines
   - Use type hints for better code clarity
   - Document all public functions and classes

## Deployment

For production deployment, consider using:

- Gunicorn or uWSGI as the WSGI server
- Nginx as a reverse proxy
- Environment variables for sensitive configuration
- Proper logging and monitoring

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
