import streamlit as st

st.set_page_config(
    page_title="About Deepfakes - Deepfake Detection Platform",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("About Deepfakes")
    
    st.markdown("""
    ## What are Deepfakes?
    
    Deepfakes are synthetic media in which a person in an existing image or video is replaced with someone else's likeness using artificial intelligence. 
    The term "deepfake" is a combination of "deep learning" and "fake".
    
    ### How Deepfakes Work
    
    Deepfakes use deep learning algorithms, particularly autoencoders and generative adversarial networks (GANs), to generate visual and audio content with a high potential to deceive.
    
    The process typically involves:
    
    1. **Collection**: Gathering image/video/audio samples of the target person
    2. **Training**: Training neural networks on these samples to learn patterns
    3. **Generation**: Creating new synthetic media that mimics the target person
    4. **Refinement**: Improving the quality and realism of the synthetic media
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://miro.medium.com/max/1400/1*uoFd7Nk3XhdAnVmjNkihYA.jpeg", 
                caption="Example of how deepfake technology works")
    
    with col2:
        st.markdown("""
        ### Types of Deepfakes
        
        - **Face Swapping**: Replacing one person's face with another in videos or images
        - **Facial Manipulation**: Changing facial expressions or making someone appear to say something they didn't
        - **Voice Synthesis**: Creating synthetic voices that sound like specific individuals
        - **Full Body Puppetry**: Manipulating someone's entire body movements
        """)
    
    st.markdown("""
    ## The Risks of Deepfakes
    
    Deepfakes pose several serious risks to individuals and society:
    
    - **Misinformation**: Creating convincing fake news or political content
    - **Fraud**: Enabling financial fraud or identity theft
    - **Harassment**: Facilitating harassment, especially through fake intimate content
    - **Erosion of Trust**: Making it harder to trust video/audio evidence
    
    ### Real-World Impact
    
    There have been numerous cases where deepfakes have been used maliciously:
    
    - Political deepfakes creating false statements by politicians
    - Celebrity deepfakes in non-consensual intimate content
    - Voice cloning for financial fraud and scams
    - Corporate sabotage through fake statements from executives
    """)
    
    st.markdown("""
    ## Detecting Deepfakes
    
    ### Technical Methods for Detection
    
    Our platform uses several advanced techniques to detect deepfakes:
    
    1. **Visual Inconsistencies**: Detecting unnatural blinking, facial asymmetry, or lighting inconsistencies
    2. **Audio Analysis**: Identifying unusual patterns in speech, breathing, or background noise
    3. **Temporal Coherence**: Analyzing consistency between frames in videos
    4. **Metadata Analysis**: Examining digital fingerprints and file information
    5. **Biological Signals**: Detecting heartbeat signals in skin color or natural head movements
    
    ### How Our Models Work
    
    Our deepfake detection models are trained on diverse datasets containing both genuine and synthetic media. The models learn to recognize subtle artifacts and inconsistencies that are often imperceptible to the human eye but consistently present in AI-generated content.
    """)
    
    st.markdown("""
    ## Protecting Yourself
    
    ### Tips to Identify Potential Deepfakes
    
    - **Verify the source**: Check if the content comes from reliable sources
    - **Look for inconsistencies**: Unnatural lighting, blinking, or facial movements
    - **Check for audio-visual sync**: Mismatched lip movements and audio
    - **Cross-reference information**: Verify claims through multiple sources
    - **Use detection tools**: Like our platform to analyze suspicious content
    
    ### Reporting Deepfakes
    
    If you encounter malicious deepfakes:
    
    1. Report to the platform where you found it
    2. Document evidence before it's removed
    3. Contact relevant authorities if it involves harassment or fraud
    4. Inform the person being impersonated if possible
    """)
    
    st.markdown("""
    ## Future of Deepfake Technology
    
    As AI continues to advance, both deepfake creation and detection technologies will evolve. This creates an ongoing technological race between creation and detection methods.
    
    Key developments to watch:
    
    - More accessible deepfake creation tools
    - Improved real-time deepfake detection
    - Blockchain-based media authentication
    - Legal and regulatory frameworks for synthetic media
    
    Our platform is committed to staying at the forefront of deepfake detection technology to help maintain trust in digital media.
    """)
    
    # Resources section
    st.subheader("Additional Resources")
    
    resources = [
        {
            "title": "The State of Deepfakes",
            "url": "https://sensity.ai/reports/",
            "description": "Reports on deepfake trends and statistics"
        },
        {
            "title": "Deepfake Detection Challenge",
            "url": "https://ai.facebook.com/datasets/dfdc/",
            "description": "Dataset and research competition for deepfake detection"
        },
        {
            "title": "Media Forensics",
            "url": "https://www.darpa.mil/program/media-forensics",
            "description": "DARPA's research program on media authentication"
        }
    ]
    
    for resource in resources:
        st.markdown(f"""
        **[{resource['title']}]({resource['url']})**  
        {resource['description']}
        """)

if __name__ == "__main__":
    main()
