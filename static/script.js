document.addEventListener('DOMContentLoaded', () => {
    const loadingSpinner = document.getElementById('loading-spinner');
    const edaSection = document.getElementById('eda');
    const mlModelsSection = document.getElementById('ml-models');

    const labelDistPlot = document.getElementById('label-distribution-plot');
    const corrHeatmapPlot = document.getElementById('correlation-heatmap-plot');
    const featureHistogramsDiv = document.getElementById('feature-histograms');

    const rfAccuracy = document.getElementById('rf-accuracy');
    const rfPrecision = document.getElementById('rf-precision');
    const rfRecall = document.getElementById('rf-recall');
    const rfF1 = document.getElementById('rf-f1');
    const rfConfusionMatrix = document.getElementById('rf-confusion-matrix');

    const lrAccuracy = document.getElementById('lr-accuracy');
    const lrPrecision = document.getElementById('lr-precision');
    const lrRecall = document.getElementById('lr-recall');
    const lrF1 = document.getElementById('lr-f1');
    const lrConfusionMatrix = document.getElementById('lr-confusion-matrix');

    // Function to display a message (instead of alert)
    const showMessage = (title, message) => {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <h2>${title}</h2>
                <p>${message}</p>
                <button class="close-button">Close</button>
            </div>
        `;
        document.body.appendChild(modal);

        modal.querySelector('.close-button').addEventListener('click', () => {
            document.body.removeChild(modal);
        });

        // Basic modal styling (add this to style.css)
        const style = document.createElement('style');
        style.innerHTML = `
            .modal {
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.6);
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .modal-content {
                background-color: #FFFFFF;
                margin: auto;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                max-width: 500px;
                text-align: center;
                color: #333;
            }
            .modal-content h2 {
                color: #800000;
                margin-top: 0;
                font-size: 1.8em;
            }
            .modal-content p {
                font-size: 1.1em;
                margin-bottom: 20px;
            }
            .close-button {
                background-color: #800000;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
                transition: background-color 0.3s ease;
            }
            .close-button:hover {
                background-color: #600000;
            }
        `;
        document.head.appendChild(style);
    };

    // Function to fetch EDA plots
    const fetchEDAPlots = async () => {
        try {
            const response = await fetch('/eda_plots');
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const plots = await response.json();

            labelDistPlot.src = `data:image/png;base64,${plots.label_distribution}`;
            corrHeatmapPlot.src = `data:image/png;base64,${plots.correlation_heatmap}`;

            // Clear previous histograms
            featureHistogramsDiv.innerHTML = '';
            for (const key in plots) {
                if (key.startsWith('hist_')) {
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${plots[key]}`;
                    img.alt = key.replace('hist_', '').replace(/_/g, ' ') + ' Distribution Plot';
                    featureHistogramsDiv.appendChild(img);
                }
            }
        } catch (error) {
            console.error('Error fetching EDA plots:', error);
            showMessage('Error', `Failed to load EDA plots: ${error.message}. Please ensure the backend is running and the dataset is correctly placed.`);
        }
    };

    // Function to fetch ML model results
    const fetchMLResults = async () => {
        try {
            const response = await fetch('/model_results');
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const results = await response.json();

            // Random Forest
            if (results.RandomForest) {
                rfAccuracy.textContent = results.RandomForest.accuracy;
                rfPrecision.textContent = results.RandomForest.precision;
                rfRecall.textContent = results.RandomForest.recall;
                rfF1.textContent = results.RandomForest.f1_score;
                rfConfusionMatrix.src = `data:image/png;base64,${results.RandomForest.confusion_matrix_plot}`;
            }

            // Logistic Regression
            if (results.LogisticRegression) {
                lrAccuracy.textContent = results.LogisticRegression.accuracy;
                lrPrecision.textContent = results.LogisticRegression.precision;
                lrRecall.textContent = results.LogisticRegression.recall;
                lrF1.textContent = results.LogisticRegression.f1_score;
                lrConfusionMatrix.src = `data:image/png;base64,${results.LogisticRegression.confusion_matrix_plot}`;
            }

        } catch (error) {
            console.error('Error fetching ML results:', error);
            showMessage('Error', `Failed to load ML model results: ${error.message}. Please ensure the backend is running.`);
        }
    };

    // Initial data load and display
    const loadDashboard = async () => {
        loadingSpinner.classList.remove('hidden'); // Show spinner
        edaSection.classList.add('hidden');
        mlModelsSection.classList.add('hidden');

        await fetchEDAPlots();
        await fetchMLResults();

        loadingSpinner.classList.add('hidden'); // Hide spinner
        edaSection.classList.remove('hidden');
        mlModelsSection.classList.remove('hidden');
    };

    loadDashboard();

    // Smooth scroll for navigation
    document.querySelectorAll('nav ul li a').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            document.querySelectorAll('nav ul li a').forEach(link => link.classList.remove('active'));
            this.classList.add('active');

            const targetId = this.getAttribute('href').substring(1);
            document.getElementById(targetId).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});