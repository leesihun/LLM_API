"""
Weather API Tool
Get current weather and forecasts
"""

import logging
import httpx
from typing import Optional, Dict, Any
from datetime import datetime


logger = logging.getLogger(__name__)


class WeatherTool:
    """
    Weather information retrieval

    Uses Open-Meteo API (free, no API key required)
    For production, can be replaced with OpenWeatherMap, WeatherAPI, etc.
    """

    def __init__(self):
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        self.timeout = 10

    async def get_weather(self, location: str) -> str:
        """
        Get current weather for a location

        Args:
            location: City name or location query

        Returns:
            Formatted weather information
        """
        logger.info(f"[Weather] Getting weather for: {location}")

        try:
            # Step 1: Geocode the location
            coordinates = await self._geocode(location)

            if not coordinates:
                return f"Could not find location: {location}"

            # Step 2: Get weather data
            weather_data = await self._fetch_weather(
                coordinates["latitude"],
                coordinates["longitude"]
            )

            if not weather_data:
                return f"Could not retrieve weather data for {location}"

            # Step 3: Format the response
            return self._format_weather(location, coordinates, weather_data)

        except Exception as e:
            logger.error(f"[Weather] Error: {e}")
            return f"Error getting weather: {str(e)}"

    async def _geocode(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Convert location name to coordinates

        Returns:
            Dict with latitude, longitude, name, country
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "name": location,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                }

                response = await client.get(self.geocoding_url, params=params)
                response.raise_for_status()

                data = response.json()

                if data.get("results"):
                    result = data["results"][0]
                    return {
                        "latitude": result["latitude"],
                        "longitude": result["longitude"],
                        "name": result["name"],
                        "country": result.get("country", ""),
                        "admin1": result.get("admin1", "")
                    }

                return None

        except Exception as e:
            logger.error(f"[Weather] Geocoding error: {e}")
            return None

    async def _fetch_weather(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        Fetch weather data from API

        Returns:
            Weather data dictionary
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m",
                    "hourly": "temperature_2m,precipitation_probability",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "precipitation_unit": "mm",
                    "timezone": "auto"
                }

                response = await client.get(self.weather_url, params=params)
                response.raise_for_status()

                return response.json()

        except Exception as e:
            logger.error(f"[Weather] Fetch error: {e}")
            return None

    def _format_weather(self, location: str, coordinates: Dict[str, Any], data: Dict[str, Any]) -> str:
        """
        Format weather data into readable text
        """
        current = data.get("current", {})
        daily = data.get("daily", {})

        # Current weather
        temp = current.get("temperature_2m", "N/A")
        feels_like = current.get("apparent_temperature", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind_speed = current.get("wind_speed_10m", "N/A")
        precipitation = current.get("precipitation", "N/A")
        weather_code = current.get("weather_code", 0)

        # Weather description from code
        weather_desc = self._get_weather_description(weather_code)

        # Format location
        location_name = coordinates["name"]
        if coordinates.get("admin1"):
            location_name += f", {coordinates['admin1']}"
        if coordinates.get("country"):
            location_name += f", {coordinates['country']}"

        # Build response
        result = f"""**Current Weather for {location_name}**

🌡️ Temperature: {temp}°C (feels like {feels_like}°C)
☁️ Conditions: {weather_desc}
💧 Humidity: {humidity}%
💨 Wind Speed: {wind_speed} km/h
🌧️ Precipitation: {precipitation} mm
"""

        # Add daily forecast if available
        if daily.get("temperature_2m_max"):
            max_temps = daily["temperature_2m_max"][:3]  # Next 3 days
            min_temps = daily["temperature_2m_min"][:3]

            result += "\n**3-Day Forecast:**\n"
            for i, (max_t, min_t) in enumerate(zip(max_temps, min_temps)):
                day_name = ["Today", "Tomorrow", "Day After"][i] if i < 3 else f"Day {i+1}"
                result += f"  {day_name}: {max_t}°C / {min_t}°C\n"

        result += f"\n📍 Coordinates: {coordinates['latitude']}, {coordinates['longitude']}"
        result += f"\n🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return result

    def _get_weather_description(self, code: int) -> str:
        """
        Convert WMO weather code to description
        """
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow fall",
            73: "Moderate snow fall",
            75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }

        return weather_codes.get(code, "Unknown")


# Global instance
weather_tool = WeatherTool()
