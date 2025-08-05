import os
import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
from utils import chunk_text

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192"
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The above text is from a job or careers page. 
            Extract all possible job postings, even if the format is unusual or incomplete.
            For each job, return a JSON object with as many of these keys as possible: 
            `company_name`, `role`, `location`, `job_type`, `experience`, `skills`, `qualifications`, `description`.
            If a field is missing, leave it blank.
            Always return a list of job objects, even if only one is found.
            If no jobs are found, return an empty list [].
            ### VALID JSON (NO PREAMBLE):
            """
        )

        all_jobs = []
        for i, chunk in enumerate(chunk_text(cleaned_text, max_tokens=500)):
            print(f"Processing chunk {i+1}, length: {len(chunk)}")
            chain_extract = prompt_extract | self.llm
            res = chain_extract.invoke(input={"page_data": chunk})
            print("LLM raw output:", res.content)

            try:
                json_parser = JsonOutputParser()
                res = json_parser.parse(res.content)
            except OutputParserException:
                continue

            if isinstance(res, list):
                valid_jobs = [job for job in res if isinstance(job, dict) and job.get('role')]
                all_jobs.extend(valid_jobs)
            elif isinstance(res, dict) and res.get('role'):
                all_jobs.append(res)

        if not all_jobs:
            raise OutputParserException(
                "No jobs could be extracted. The page may not contain job listings, or the format is not supported. Try a different link, or paste the job description manually."
            )

        return all_jobs

    def extract_resume_details(self, resume_text):
        prompt_resume = PromptTemplate.from_template(
            """
            ### RESUME TEXT:
            {resume_text}

            ### INSTRUCTION:
            Parse the provided resume text and return a structured JSON format containing the following keys: 
            `name`, `email`, `phone_number`, `address`, `education`, `skills`, `experience`, `projects`, `extra_curricular`, and `committees_and_clubs`.
            Ensure that all sections are captured accurately and concisely. The `address` field is optional and should be included only if present.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_resume = prompt_resume | self.llm
        res = chain_resume.invoke({"resume_text": resume_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse resume details.")

        return res if isinstance(res, dict) else {}

    def write_cover_letter(self, job, resume_info):
        date = datetime.date.today().strftime("%B %d, %Y")
        prompt_cover_letter = PromptTemplate.from_template(
            """
            You're a cover letter writing expert. Use the details and instructions below to write a professional and tailored cover letter for a job application.

            ### JOB DESCRIPTION: (job at {company_name} for the {role} position)
            {job_description}

            ### CANDIDATE'S RESUME INFORMATION:
            Name: {name}
            Email: {email}
            Phone Number: {phone_number}
            Address: {address}
            Education: {education}
            Skills: {skills}
            Experience: {experience}
            Projects: {projects}
            Extracurricular Activities: {extra_curricular}
            Committees and Clubs: {committees_and_clubs}

            ### INSTRUCTION:
            1. Start with the candidate's contact information at the top, followed by the date (aligned left or right).
            2. Add the recipient's address (e.g., "To The Hiring Manager," company name, company address if available).
            3. Insert a clear subject line, e.g., "Subject: Application for the {role} Position".
            4. Use a professional salutation (e.g., "Dear Hiring Manager,").
            5. Write the body in 2-3 concise, well-separated paragraphs:
                - First paragraph: State the position, how you found it, and a brief introduction.
                - Second paragraph: Highlight your most relevant skills, experience, and achievements that match the job requirements. Use concrete examples.
                - Third paragraph: Express enthusiasm for the company, why you are a good fit, and a polite closing.
            6. End with a formal closing (e.g., "Sincerely,") and the candidate's name.
            7. Ensure clarity, professionalism, and readability. Avoid dense text—use paragraph breaks.
            8. Do not include any preamble or explanation—output only the cover letter.
            ### COVER LETTER (NO PREAMBLE):
            """
        )

        chain_cover_letter = prompt_cover_letter | self.llm
        res = chain_cover_letter.invoke({
            "job_description": str(job),
            "company_name": job.get('company_name'),
            "role": job.get('role'),
            "name": resume_info.get('name'),
            "email": resume_info.get('email', 'No email provided.'),
            "phone_number": resume_info.get('phone_number', 'No phone number provided.'),
            "address": resume_info.get('address', 'No address provided.'),
            "education": resume_info.get('education'),
            "skills": resume_info.get('skills'),
            "experience": resume_info.get('experience'),
            "projects": resume_info.get('projects'),
            "extra_curricular": resume_info.get('extra_curricular'),
            "committees_and_clubs": resume_info.get('committees_and_clubs'),
            "date": date
        })
        return res.content

    def chat_with_llm(self, message, resume_info=None, job_info=None):
        prompt_chat = PromptTemplate.from_template(
            """
            ### USER QUESTION:
            {user_message}

            ### RESUME INFORMATION (OPTIONAL):
            {resume_info}

            ### JOB INFORMATION (OPTIONAL):
            {job_info}

            ### INSTRUCTION:
            Provide an intelligent and helpful response to the user’s query. 
            If resume and job info are provided, tailor the response based on that.
            ### RESPONSE:
            """
        )
        chain_chat = prompt_chat | self.llm
        res = chain_chat.invoke({
            "user_message": message,
            "resume_info": resume_info or "No resume info provided.",
            "job_info": job_info or "No job info provided."
        })
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
