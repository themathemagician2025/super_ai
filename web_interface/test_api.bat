@echo off
echo Testing API endpoints...

echo.
echo 1. Testing /api/scraping/status
curl http://localhost:5001/api/scraping/status
echo.

echo 2. Testing /api/scraping/start
curl -X POST -H "Content-Type: application/json" -d "{\"sources\":[\"sports\"]}" http://localhost:5001/api/scraping/start
echo.

echo 3. Testing /api/predict
curl -X POST -H "Content-Type: application/json" -d "{\"type\":\"sports\",\"home_team\":\"Team A\",\"away_team\":\"Team B\"}" http://localhost:5001/api/predict
echo.

echo 4. Testing /api/query
curl -X POST -H "Content-Type: application/json" -d "{\"data_type\":\"sports\",\"limit\":5}" http://localhost:5001/api/query
echo.

echo 5. Testing /api/training/start
curl -X POST -H "Content-Type: application/json" -d "{\"model_type\":\"sports\"}" http://localhost:5001/api/training/start
echo.

echo 6. Testing /api/training/status
curl http://localhost:5001/api/training/status
echo.

echo All tests completed.
pause
