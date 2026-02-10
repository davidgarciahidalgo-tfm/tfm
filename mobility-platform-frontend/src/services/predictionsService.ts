import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export interface HourlyPrediction {
  hour: string
  devices: number
}

export interface StationPrediction {
  stationId: number
  stationName: string
  currentDevices: number
  predictedDevices: number
  change: number
  changePercent: number
  confidence: 'high' | 'medium' | 'low'
  hourlyPredictions: HourlyPrediction[]
}

export interface PredictionsResponse {
  date: string
  predictions: StationPrediction[]
  modelInfo: {
    name: string
    version: string
    lastTrained: string
    accuracy: number
  }
}

/**
 * Obtiene las predicciones del modelo VAR para el día siguiente
 */
export const getDailyPredictions = async (): Promise<PredictionsResponse> => {
  try {
    const response = await axios.get<PredictionsResponse>(`${API_BASE_URL}/predictions/daily`)
    return response.data
  } catch (error) {
    console.error('Error fetching predictions:', error)
    throw error
  }
}

/**
 * Obtiene predicciones para una estación específica
 */
export const getStationPrediction = async (stationId: number): Promise<StationPrediction> => {
  try {
    const response = await axios.get<StationPrediction>(`${API_BASE_URL}/predictions/station/${stationId}`)
    return response.data
  } catch (error) {
    console.error(`Error fetching prediction for station ${stationId}:`, error)
    throw error
  }
}

/**
 * Obtiene predicciones para múltiples días
 */
export const getMultiDayPredictions = async (days: number = 7): Promise<PredictionsResponse[]> => {
  try {
    const response = await axios.get<PredictionsResponse[]>(`${API_BASE_URL}/predictions/multi-day`, {
      params: { days }
    })
    return response.data
  } catch (error) {
    console.error('Error fetching multi-day predictions:', error)
    throw error
  }
}

/**
 * Obtiene información sobre el modelo VAR
 */
export const getModelInfo = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/predictions/model-info`)
    return response.data
  } catch (error) {
    console.error('Error fetching model info:', error)
    throw error
  }
}

/**
 * Solicita un re-entrenamiento del modelo (solo admin)
 */
export const retrainModel = async () => {
  try {
    const response = await axios.post(`${API_BASE_URL}/predictions/retrain`)
    return response.data
  } catch (error) {
    console.error('Error retraining model:', error)
    throw error
  }
}

export default {
  getDailyPredictions,
  getStationPrediction,
  getMultiDayPredictions,
  getModelInfo,
  retrainModel
}
