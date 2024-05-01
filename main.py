import os
import base64
import streamlit as st
from utils import Test
from PIL import Image
#from Test import get_pdf_text,load_llm,add_graph_db,search

o1=Test()

def user_input(user_question):
    llm = o1.load_llm()
    chain=o1.search(llm)
    result=chain.invoke({"question":user_question})
    st.write("Answer: ", result)
    
def main():
    img=Image.open("C:\\Users\\POAGRAWA\\Downloads\\front.jpg")
    st.set_page_config(layout="centered",page_title="Quick & efficient retreival of context aware search results and relevant information",page_icon=img)
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
    #st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.header(":blue[Knowledge mining of Engineering documents] :red[&] :blue[Streamline Employee Onboarding Process] :red[!!]",divider='rainbow')
    #st.header('_Streamlit_ is :blue[cool] :sunglasses:')
    label = "How I can help you?💁"
    user_question = st.chat_input(label)
    #user_question=st.text_input(r"$\textsf{\large How I can help you?💁}$")
    if user_question:
        st.write("Question: ", user_question)
        with st.spinner("Processing..."):
            user_input(user_question)  # Handle user input and start a conversation
    # Define a function to convert binary file to base64
    @st.cache_data
    def get_base64(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    # Set a background image for the app
    def set_background(png_file):
        bin_str = get_base64(png_file)
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/jpg;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)
    #set_background('background4.jpg')
    with st.sidebar:
        st.title(":red[Upload latest documents for Search:] :open_book:")
        pdf_docs = st.file_uploader("Upload the file in PDF format and Click on the Submit & Process Button", accept_multiple_files=True,type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                final_pdf_docs=[]
                for file in pdf_docs:
                    if file.name.endswith(('.pdf', '.PDF')):
                       final_pdf_docs.append(file)
                    else:
                        st.error(f"File format is not supported for uploaded file: {file.name}", icon="🚨")
                print(final_pdf_docs)
                if final_pdf_docs:
                    for pdffile in final_pdf_docs:
                        with open(os.path.join("tmpDir",pdffile.name),"wb") as f:
                            print(f"--------------------------------------------{f}-------------------------------")
                            f.write(pdffile.getbuffer())
                            st.success(f"Saved File: {pdffile.name} to tmpDir")
                    documents=o1.get_pdf_text(final_pdf_docs)
                    llm = o1.load_llm()
                    o1.add_graph_db(llm,documents)
                    st.success("Upload is successfully completed. Kindly ask the questions now!!",icon="✅")
if __name__ == "__main__":
    main()

   