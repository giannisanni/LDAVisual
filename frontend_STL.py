import streamlit as st
import streamlit.components.v1 as components

# Set page configuration with the sidebar always expanded
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# Function to read and display an HTML file responsively
def display_html(file_path):
    with open(file_path, 'r') as f:
        html_data = f.read()
        # Use 100% width for responsiveness within the Streamlit container
        components.html(html_data, height=800, width=1200, scrolling=True)

# Define the sidebar for navigation
with st.sidebar:
    page = st.radio("Choose a visualization set:", ["All DG", "Seperated DG", "All Data"])

# Main Streamlit app
def main():
    if page == "All DG":
        st.title("HTML File Visualizations - combined visualization of 5 DG's")

        # Tab layout for Set 1
        tab1, tab2, tab3, tab4 = st.tabs(["10 Topics", "12 Topics", "15 Topics", "20 Topics"])

        with tab1:
            st.header("Visualization 1")
            display_html('ldavis_prepared_10CombinedALL_10.html')

        with tab2:
            st.header("Visualization 2")
            display_html('ldavis_prepared_12CombinedALL_12.html')

        with tab3:
            st.header("Visualization 3")
            display_html('ldavis_prepared_15CombinedALL_15.html')

        with tab4:
            st.header("Visualization 4")
            display_html('ldavis_prepared_20CombinedALL_20.html')

    elif page == "Seperated DG":
        st.title("HTML File Visualizations")

        # Tab layout for Set 2
        tab5, tab6, tab7, tab8, tab9 = st.tabs(["WB10T", "RWS10T", "MeI10T", "MB10T", "LeM10T"])

        with tab5:
            st.header("Visualization 1")
            display_html('ldavis_prepared_10WB.html')

        with tab6:
            st.header("Visualization 2")
            display_html('ldavis_prepared_10RWS.html')

        with tab7:
            st.header("Visualization 3")
            display_html('ldavis_prepared_10MeI.html')

        with tab8:
            st.header("Visualization 4")
            display_html('ldavis_prepared_10MB.html')

        with tab9:
            st.header("Visualization 5")
            display_html('ldavis_prepared_10LeM.html')

    elif page == "All Data":
        st.title("HTML File Visualizations")

        # Tab layout for Set 2
        tab10, tab11, tab12 = st.tabs(["10T", "15T", "20T"])

        with tab10:
            st.header("Visualization 1")
            display_html('ldavis_prepared_10ALLdata.html')

        with tab11:
            st.header("Visualization 2")
            display_html('ldavis_prepared_15ALLdata.html')

        with tab12:
            st.header("Visualization 3")
            display_html('ldavis_prepared_20ALLdata.html')

if __name__ == "__main__":
    main()


