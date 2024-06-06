import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2, l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the URLs to the datasets
dataset_urls = {
    "U-Shape": r"C:\Users\CHARISHMA\Downloads\Multiple CSV\Multiple CSV\1.ushape.csv",
    "Conecntric Circle-1": r"C:\Users\CHARISHMA\Downloads\Multiple CSV\Multiple CSV\2.concerticcir1.csv",
    "Conecntric Circle-2": r"C:\Users\CHARISHMA\Downloads\Multiple CSV\Multiple CSV\3.concertriccir2.csv",
    "Linear": r"C:\Users\CHARISHMA\Downloads\Multiple CSV\Multiple CSV\4.linearsep.csv",
    "Outlier": r"C:\Users\CHARISHMA\Downloads\Multiple CSV\Multiple CSV\5.outlier.csv",
    "Overlap": r"C:\Users\CHARISHMA\Downloads\Multiple CSV\Multiple CSV\6.overlap.csv",
    "Xor": r"C:\Users\CHARISHMA\Downloads\Multiple CSV\Multiple CSV\7.xor.csv",
    "Two Spirals": r"C:\Users\CHARISHMA\Downloads\Multiple CSV\Multiple CSV\8.twospirals.csv",
}

# Custom CSS for sidebar and main content colors
st.markdown("""
    <style>
        .css-1d391kg {  /* Class for sidebar */
            background-color: #42b6f5; /* Sidebar background color */
        }
        .css-18e3th9 {  /* Class for main content */
            background-color: #f8f9fa; /* Main content background color */
        }
    </style>
""", unsafe_allow_html=True)

# Application heading
st.title("TensorFlow Playground")

# Sidebar options for dataset selection
st.sidebar.title("Dataset and Model Configuration")
selected_dataset = st.sidebar.selectbox("Select Dataset", list(dataset_urls.keys()))

# Sidebar options for model configuration
num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 1)

layer_configurations = []
for i in range(num_hidden_layers):
    layer_config = {}
    layer_config['num_neurons'] = st.sidebar.slider(f"Number of Neurons for Layer {i+1}", 1, 50, 32)
    layer_config['activation_function'] = st.sidebar.selectbox(f"Activation Function for Layer {i+1}", ["relu", "sigmoid", "tanh"], key=f"activation_{i}")
    layer_configurations.append(layer_config)

learning_rate = st.sidebar.selectbox("Learning Rate", [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=1000, value=100, step=10)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=512, value=32, step=1)
problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])
regularization_type = st.sidebar.selectbox("Regularization Type", ["None", "L1", "L2", "L1_L2"])
regularization_factor = st.sidebar.number_input("Regularization Factor", min_value=0.0, value=0.01, step=0.001)

# Load the selected dataset
@st.cache_data
def load_data(url):
    try:
        data = pd.read_csv(url, header=None)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

data_url = dataset_urls[selected_dataset]
data = load_data(data_url)

if data is not None:
    st.write(f"### {selected_dataset}")
    st.write(data.head())

    # Assume the last column is the target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    if X.shape[1] != 2:
        st.write("This visualization works only for 2D datasets.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if st.sidebar.button('Submit'):
            inputs = Input(shape=(X_train.shape[1],))
            regularizer = None
            if regularization_type == "L1":
                regularizer = l1(regularization_factor)
            elif regularization_type == "L2":
                regularizer = l2(regularization_factor)
            elif regularization_type == "L1_L2":
                regularizer = l1_l2(regularization_factor)

            x = Dense(layer_configurations[0]['num_neurons'], activation=layer_configurations[0]['activation_function'], kernel_regularizer=regularizer)(inputs)
            for i in range(1, num_hidden_layers):
                x = Dense(layer_configurations[i]['num_neurons'], activation=layer_configurations[i]['activation_function'], kernel_regularizer=regularizer)(x)
            
            if problem_type == "Classification":
                outputs = Dense(1, activation='sigmoid')(x)
                loss_function = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                outputs = Dense(1, activation='linear')(x)
                loss_function = 'mean_squared_error'
                metrics = ['mse']

            model = Model(inputs, outputs)
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=metrics)
            history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

            st.write("### Model Summary")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))

            class KerasClassifierWrapper:
                def __init__(self, model):
                    self.model = model
                def predict(self, X):
                    predictions = self.model.predict(X)
                    return (predictions > 0.5).astype(int) if problem_type == "Classification" else predictions

            st.write("### Decision Regions for Each Neuron")
            total_neurons = sum(config['num_neurons'] for config in layer_configurations)
            fig, axes = plt.subplots((total_neurons + 1) // 2, 2, figsize=(14, 7 * ((total_neurons + 1) // 2)))
            axes = axes.flatten()

            X_combined = np.vstack((X_train, X_test))
            y_combined = np.hstack((y_train, y_test))

            neuron_index = 0
            for layer_index, layer_config in enumerate(layer_configurations):
                for neuron_index_in_layer in range(layer_config['num_neurons']):
                    partial_model = Model(inputs=inputs, outputs=model.layers[layer_index+1].output[:, neuron_index_in_layer])
                    ax = axes[neuron_index]
                    plot_decision_regions(X_combined, y_combined.astype(int), clf=KerasClassifierWrapper(partial_model), ax=ax, legend=2)
                    ax.set_title(f'Layer {layer_index+1}, Neuron {neuron_index_in_layer+1}')
                    neuron_index += 1
            st.pyplot(fig)

            st.write("### Final Decision Region")
            plt.figure(figsize=(14, 7))
            plot_decision_regions(X_combined, y_combined.astype(int), clf=KerasClassifierWrapper(model), legend=2)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Final Decision Region')
            st.pyplot(plt)

            st.write("### Training and Validation Loss")
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label="Train Loss")
            plt.plot(history.history['val_loss'], label="Validation Loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            st.pyplot(plt)
