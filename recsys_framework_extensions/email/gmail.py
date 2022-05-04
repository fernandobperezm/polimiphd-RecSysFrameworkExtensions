import base64
import os
from email.mime.text import MIMEText

import attrs
import toml
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient import errors
from googleapiclient.discovery import build, Resource


@attrs.define(kw_only=True, frozen=True)
class GmailEmailNotifierConfig:
    path_credential_json: str = attrs.field()
    path_token_json: str = attrs.field()
    gmail_scopes: list[str] = attrs.field(
        default=[
            'https://www.googleapis.com/auth/gmail.readonly',
            "https://www.googleapis.com/auth/gmail.compose",
        ]
    )


def _load_config() -> GmailEmailNotifierConfig:
    path_pyproject_toml = os.path.join(os.getcwd(), "pyproject.toml")

    with open(path_pyproject_toml, "r") as pyproject_toml_file:
        config = toml.load(f=pyproject_toml_file)

    logs_config = GmailEmailNotifierConfig(
        **(config["email"]["gmail"])
    )
    return logs_config


def _setup_email_client() -> Resource:
    """
    Extracted from: https://developers.google.com/gmail/api/quickstart/python
    """
    config = _load_config()

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(config.path_token_json):
        creds = Credentials.from_authorized_user_file(
            filename=config.path_token_json,
            scopes=config.gmail_scopes,
        )

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file=config.path_credential_json,
                scopes=config.gmail_scopes,
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(config.path_token_json, 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service


class GmailEmailNotifier:
    # If modifying these scopes, delete the file token.json.
    @staticmethod
    def send_email(
        subject: str,
        body: str,
        sender: str,
        receivers: list[str],
    ) -> None:
        assert len(subject) > 0
        assert len(body) > 0
        assert len(sender) > 0
        assert len(receivers) > 0

        str_receivers = ",".join(receivers)

        message = MIMEText(body)
        message['to'] = str_receivers
        message['from'] = sender
        message['subject'] = subject

        obj_message = {
            'raw': base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode(encoding="UTF-8"),
        }

        try:
            email_client = _setup_email_client()
            message_result = (
                email_client
                    .users()
                    .messages()
                    .send(userId="me", body=obj_message)
                    .execute()
            )
            print('Message Id: %s' % message_result['id'])
            return message_result
        except errors.HttpError as error:
            print('An error occurred: %s' % error)
        except:
            return
