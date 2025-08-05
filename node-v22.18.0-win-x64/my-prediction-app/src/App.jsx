import { useState } from 'react';
import { Shield, FlaskConical, CircleAlert } from 'lucide-react';

const App = () => {
    // State to manage form inputs
    const [formData, setFormData] = useState({
        age: 45,
        gender: 'Male',
        smoker: 'Yes',
        yearsSmoking: 10,
        cigsPerDay: 5,
    });

    // State for the prediction result
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    // Function to handle changes in form inputs
    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prevState => ({
            ...prevState,
            [name]: value,
        }));
    };

    // Function to simulate a prediction. In a real app, this would call a backend API.
    const handlePredict = async () => {
        // Start loading state
        setLoading(true);
        setPrediction(null);

        // Simulate an API call with a delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        // Get the form data
        const { age, smoker, yearsSmoking, cigsPerDay } = formData;

        // Simulate prediction logic based on a simplified heuristic
        // This is not a real model, but a demonstration of how the UI would work.
        let riskScore = 0;

        // Higher risk for older individuals
        if (age > 60) riskScore += 3;
        else if (age > 45) riskScore += 1;

        // Higher risk for smokers
        if (smoker === 'Yes') {
            riskScore += 2;
            // Additional risk based on smoking duration and intensity
            if (yearsSmoking > 20) riskScore += 3;
            else if (yearsSmoking > 10) riskScore += 1;

            if (cigsPerDay > 15) riskScore += 2;
            else if (cigsPerDay > 5) riskScore += 1;
        }

        // Determine the prediction result based on the risk score
        if (riskScore > 5) {
            setPrediction({
                status: 'High Risk',
                message: 'Based on the provided data, the risk of lung cancer is estimated to be high.',
                icon: <Shield className="h-10 w-10 text-red-500" />,
                color: 'bg-red-100 border-red-500 text-red-800'
            });
        } else if (riskScore > 2) {
            setPrediction({
                status: 'Moderate Risk',
                message: 'Based on the provided data, the risk of lung cancer is estimated to be moderate.',
                icon: <FlaskConical className="h-10 w-10 text-yellow-500" />,
                color: 'bg-yellow-100 border-yellow-500 text-yellow-800'
            });
        } else {
            setPrediction({
                status: 'Low Risk',
                message: 'Based on the provided data, the risk of lung cancer is estimated to be low.',
                icon: <CircleAlert className="h-10 w-10 text-green-500" />,
                color: 'bg-green-100 border-green-500 text-green-800'
            });
        }

        // End loading state
        setLoading(false);
    };

    return (
        <div className="bg-gray-100 min-h-screen p-8 flex items-center justify-center font-sans">
            <div className="w-full max-w-2xl bg-white p-8 rounded-2xl shadow-xl border border-gray-200">
                <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-2 text-center">
                    Lung Cancer Risk Predictor
                </h1>
                <p className="text-center text-gray-500 mb-8">
                    Enter patient data to get a risk assessment.
                </p>

                {/* Input Form */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                    {/* Age Input */}
                    <div className="flex flex-col">
                        <label htmlFor="age" className="text-sm font-medium text-gray-700 mb-1">Age</label>
                        <input
                            type="number"
                            id="age"
                            name="age"
                            value={formData.age}
                            onChange={handleInputChange}
                            className="p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200"
                        />
                    </div>

                    {/* Gender Select */}
                    <div className="flex flex-col">
                        <label htmlFor="gender" className="text-sm font-medium text-gray-700 mb-1">Gender</label>
                        <select
                            id="gender"
                            name="gender"
                            value={formData.gender}
                            onChange={handleInputChange}
                            className="p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200"
                        >
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>

                    {/* Smoker Select */}
                    <div className="flex flex-col">
                        <label htmlFor="smoker" className="text-sm font-medium text-gray-700 mb-1">Smoker Status</label>
                        <select
                            id="smoker"
                            name="smoker"
                            value={formData.smoker}
                            onChange={handleInputChange}
                            className="p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200"
                        >
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    
                    {/* Years of Smoking Input (conditionally rendered) */}
                    {formData.smoker === 'Yes' && (
                        <div className="flex flex-col">
                            <label htmlFor="yearsSmoking" className="text-sm font-medium text-gray-700 mb-1">Years of Smoking</label>
                            <input
                                type="number"
                                id="yearsSmoking"
                                name="yearsSmoking"
                                value={formData.yearsSmoking}
                                onChange={handleInputChange}
                                className="p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200"
                            />
                        </div>
                    )}

                    {/* Cigarettes per Day Input (conditionally rendered) */}
                    {formData.smoker === 'Yes' && (
                        <div className="flex flex-col">
                            <label htmlFor="cigsPerDay" className="text-sm font-medium text-gray-700 mb-1">Cigarettes per Day</label>
                            <input
                                type="number"
                                id="cigsPerDay"
                                name="cigsPerDay"
                                value={formData.cigsPerDay}
                                onChange={handleInputChange}
                                className="p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200"
                            />
                        </div>
                    )}
                </div>

                {/* Prediction Button */}
                <div className="flex justify-center">
                    <button
                        onClick={handlePredict}
                        className="w-full md:w-1/2 bg-blue-600 text-white font-bold py-3 px-6 rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition duration-200 transform hover:scale-105"
                        disabled={loading}
                    >
                        {loading ? 'Predicting...' : 'Get Prediction'}
                    </button>
                </div>

                {/* Prediction Result Display */}
                {prediction && (
                    <div className={`mt-8 p-6 rounded-2xl shadow-inner border-2 flex items-center space-x-4 transition-all duration-500 ${prediction.color}`}>
                        <div className="flex-shrink-0">
                            {prediction.icon}
                        </div>
                        <div>
                            <p className="text-lg font-bold mb-1">{prediction.status}</p>
                            <p className="text-sm">{prediction.message}</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default App;
