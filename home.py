import streamlit as st
import mne
import numpy as np
import pandas as pd
import tempfile
import base64
import io
from datetime import datetime
from mne.time_frequency import psd_array_welch
import plotly.graph_objects as go

st.set_page_config(page_title="EEG Analyzer - Emotiv", layout="wide")
st.title("🧠 Emotiv EEG Analyzer & Dashboard")

page = st.sidebar.radio("📁 Navigation", ["Upload & Analyze", "Dashboard", "Details & Download"])
session_state = st.session_state

if "df_results" not in session_state:
    session_state.df_results = None

# --- Upload and Analyze Page ---
if page == "Upload & Analyze":
    uploaded_file = st.file_uploader("📤 Upload a `.fif` or `.edf` EEG File", type=["fif", "edf"])

    if uploaded_file:
        suffix = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            eeg_path = tmp_file.name

        with st.spinner("🔄 Processing EEG data..."):
            if suffix == "fif":
                raw = mne.io.read_raw_fif(eeg_path, preload=True)
            elif suffix == "edf":
                raw = mne.io.read_raw_edf(eeg_path, preload=True)
            else:
                st.error("Unsupported file format.")
                st.stop()

            raw.pick_types(eeg=True)
            raw.filter(1., 45., fir_design='firwin')

            # Select only standard 14 Emotiv channels
            emotiv_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                               'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
            available_channels = [ch for ch in emotiv_channels if ch in raw.ch_names]
            raw.pick_channels(available_channels)

            data = raw.get_data()
            sfreq = raw.info['sfreq']
            channel_names = raw.ch_names

            bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30),
                'gamma': (30, 45)
            }

            psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=1, fmax=45, n_fft=1024, average='mean')
            total_power = np.sum(psds, axis=1)

            results = []
            for ch_idx, ch_name in enumerate(channel_names):
                ch_result = {"channel": ch_name}
                for band, (fmin, fmax) in bands.items():
                    freq_mask = (freqs >= fmin) & (freqs <= fmax)
                    band_power = np.sum(psds[ch_idx, freq_mask])
                    rel_power = band_power / total_power[ch_idx]
                    ch_result[f"{band}_abs"] = band_power
                    ch_result[f"{band}_rel"] = rel_power
                results.append(ch_result)

            df_results = pd.DataFrame(results)
            session_state.df_results = df_results

        st.success("✅ EEG analysis complete!")
        st.dataframe(session_state.df_results, use_container_width=True)

# --- Helper for Regional Processing ---
def compute_region_averages(df, regions):
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    result = []
    for band in bands:
        row = {'band': band}
        for region, chs in regions.items():
            subset = df[df['channel'].isin(chs)]
            abs_col = f"{band}_abs"
            rel_col = f"{band}_rel"
            row[f"{region}_abs"] = subset[abs_col].mean() if not subset.empty else 0
            row[f"{region}_rel"] = subset[rel_col].mean() if not subset.empty else 0
        result.append(row)
    return pd.DataFrame(result)

# --- Dashboard View ---
if page == "Dashboard" and session_state.df_results is not None:
    df = session_state.df_results.copy()

    # Only show non-delta bands for clarity
    band_cols = ['theta_abs', 'alpha_abs', 'beta_abs', 'gamma_abs']

    region_map = {
        "Left": ['AF3', 'F3', 'FC5', 'T7', 'P7', 'O1'],
        "Right": ['AF4', 'F4', 'FC6', 'T8', 'P8', 'O2'],
        "Front": ['AF3', 'AF4', 'F3', 'F4'],
        "Back": ['O1', 'O2', 'P7', 'P8']
    }

    region_df = compute_region_averages(df, region_map)
    region_df = region_df[region_df['band'] != 'delta']

    # --- Normalized Line Chart (Fake Time-Series) ---
    st.subheader("📈 Normalized Brain Wave Trends (0–1 per band)")
    line_data = pd.DataFrame({'Time': pd.date_range("2025-06-18 15:10", periods=300, freq='1S')})

    for band in band_cols:
        band_data = df[band]
        normalized = (band_data - band_data.min()) / (band_data.max() - band_data.min())
        noise = np.random.normal(0, 0.05, 300)
        line_data[band] = normalized.mean() + noise

    fig_line = go.Figure()
    color_map = {
        'theta_abs': 'purple',
        'alpha_abs': 'green',
        'beta_abs': 'blue',
        'gamma_abs': 'orange'
    }
    for band in band_cols:
        fig_line.add_trace(go.Scatter(
            x=line_data['Time'], y=line_data[band],
            mode='lines', name=band.replace('_abs', '').capitalize(),
            line=dict(color=color_map[band])
        ))
    fig_line.update_layout(xaxis_title="Time", yaxis_title="Normalized Power (0–1)", height=400)
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Left vs Right and Front vs Back ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Left vs Right (Absolute Power)")
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Bar(
            name='Left', 
            y=region_df['band'], 
            x=-region_df['Left_abs'],  # LEFT is NEGATIVE
            marker_color='blue', 
            orientation='h'
        ))
        fig_lr.add_trace(go.Bar(
            name='Right', 
            y=region_df['band'], 
            x=region_df['Right_abs'],  # RIGHT is POSITIVE
            marker_color='red', 
            orientation='h'
        ))
        fig_lr.update_layout(
            barmode='relative',
            title="Session Average - Left vs Right",
            height=350,
            xaxis_title="Power (µV²/Hz)",
            xaxis=dict(zeroline=True),
            showlegend=True
        )


        xaxis=dict(autorange='reversed')
        st.plotly_chart(fig_lr, use_container_width=True)

    with col2:
        st.markdown("### 📊 Front vs Back (Absolute Power)")
        fig_fb = go.Figure()
        fig_fb.add_trace(go.Bar(
            name='Front', 
            y=region_df['band'], 
            x=-region_df['Front_abs'],  # FRONT = negative = left
            marker_color='blue', 
            orientation='h'
        ))
        fig_fb.add_trace(go.Bar(
            name='Back', 
            y=region_df['band'], 
            x=region_df['Back_abs'],  # BACK = positive = right
            marker_color='red', 
            orientation='h'
        ))
        fig_fb.update_layout(
            barmode='relative',
            title="Session Average - Front vs Back",
            height=350,
            xaxis_title="Power (µV²/Hz)",
            xaxis=dict(zeroline=True),
            showlegend=True
        )

        st.plotly_chart(fig_fb, use_container_width=True)

    # --- Gauges ---
    st.markdown("### 🕹️ Average Band Power Gauges")
    band_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    cols = st.columns(5)
    for i, band in enumerate(band_labels):
        with cols[i]:
            avg_abs = df[f"{band.lower()}_abs"].mean()
            avg_rel = df[f"{band.lower()}_rel"].mean()
            st.markdown(f"**{band}**")
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_abs,
                title={'text': "Abs"},
                gauge={'axis': {'range': [None, avg_abs * 2]}}
            )), use_container_width=True, key=f"{band}_abs")

            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_rel * 100,
                number={'suffix': '%', 'valueformat': '.1f'},
                title={'text': "Rel"},
                gauge={'axis': {'range': [0, 100]}}
            )), use_container_width=True, key=f"{band}_rel")

elif page == "Details & Download" and session_state.df_results is not None:
    df = session_state.df_results.copy()
    st.subheader("📋 EEG Per Channel - Absolute and Relative Power")

    st.markdown("#### 📌 Absolute Power")
    pivot_abs = df[['channel', 'delta_abs', 'theta_abs', 'alpha_abs', 'beta_abs', 'gamma_abs']]
    st.dataframe(pivot_abs.set_index('channel').style.format("{:.8f}"), use_container_width=True)

    st.markdown("#### 📌 Relative Power (Full Precision)")
    pivot_rel = df[['channel', 'delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel', 'gamma_rel']].copy()
    preview_rel = pivot_rel[['channel', 'delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel', 'gamma_rel']].copy()
    preview_rel.columns = ['channel', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    st.dataframe(preview_rel, use_container_width=True)

    name = st.text_input("Input your name for filename:")
    if name.strip():
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_abs = pivot_abs.to_csv(index=False).encode('utf-8')
        filename_abs = f"{name}_OtaQku_Emotiv_Absolute_{now}.csv"
        st.download_button("📥 Download Absolute Power CSV", data=csv_abs, file_name=filename_abs, mime='text/csv')

        csv_rel = preview_rel.to_csv(index=False).encode('utf-8')
        filename_rel = f"{name}_OtaQku_Emotiv_Relative_{now}.csv"
        st.download_button("📥 Download Relative Power CSV (Full Precision)", data=csv_rel, file_name=filename_rel, mime='text/csv')
    else:
        st.warning("⚠️ Please input your name to enable download buttons.")
