#!/usr/bin/env python3
"""
Google Drive 업로드 테스트 스크립트
"""

import os
import json
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    print("❌ 필요한 라이브러리가 설치되지 않았습니다.")
    print("pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2")
    sys.exit(1)

from config import DRIVE_FOLDER_NAME, DRIVE_SHARE_EMAIL

SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_PATH = "gspread-service-account.json"

def test_upload():
    print("=" * 60)
    print("Google Drive 업로드 테스트 시작")
    print("=" * 60)

    # 1. 인증 파일 확인
    if not os.path.exists(CREDENTIALS_PATH):
        print(f"❌ 인증 파일({CREDENTIALS_PATH})이 없습니다.")
        return

    try:
        creds_data = json.load(open(CREDENTIALS_PATH))
        service_email = creds_data.get("client_email", "Unknown")
        print(f"🔑 서비스 계정 이메일: {service_email}")
        
        if "private_key" not in creds_data or "BEGIN PRIVATE KEY" not in creds_data["private_key"]:
            print("❌ 인증 파일 내용이 올바르지 않습니다 (Placeholder일 가능성).")
            return
            
    except Exception as e:
        print(f"❌ 인증 파일 읽기 오류: {e}")
        return

    # 2. Drive API 연결
    try:
        creds = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
        service = build("drive", "v3", credentials=creds)
        print("✅ Drive API 연결 성공")
    except Exception as e:
        print(f"❌ Drive API 연결 실패: {e}")
        return

    # 3. 폴더 검색 또는 생성
    folder_id = None
    query = (
        f"mimeType='application/vnd.google-apps.folder' and "
        f"name='{DRIVE_FOLDER_NAME}' and trashed=false"
    )
    
    try:
        results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
        files = results.get("files", [])
        
        if files:
            folder_id = files[0]["id"]
            print(f"✅ 폴더 찾음: {files[0]['name']} (ID: {folder_id})")
        else:
            print(f"ℹ️ 폴더 '{DRIVE_FOLDER_NAME}'가 없어 새로 생성합니다.")
            folder_metadata = {
                "name": DRIVE_FOLDER_NAME,
                "mimeType": "application/vnd.google-apps.folder",
            }
            folder = service.files().create(body=folder_metadata, fields="id").execute()
            folder_id = folder.get("id")
            print(f"✅ 폴더 생성 완료 (ID: {folder_id})")
            
            # 공유 설정
            if DRIVE_SHARE_EMAIL:
                print(f"ℹ️ {DRIVE_SHARE_EMAIL} 계정으로 공유 시도...")
                try:
                    user_permission = {
                        "type": "user",
                        "role": "writer",
                        "emailAddress": DRIVE_SHARE_EMAIL,
                    }
                    service.permissions().create(
                        fileId=folder_id, body=user_permission, fields="id"
                    ).execute()
                    print(f"✅ 공유 성공")
                except Exception as e:
                    print(f"⚠️ 공유 실패: {e}")

    except Exception as e:
        print(f"❌ 폴더 검색/생성 실패 (권한 문제일 수 있음): {e}")
        return

    # 4. 파일 업로드 테스트
    test_file_name = "test_upload.txt"
    with open(test_file_name, "w") as f:
        f.write("Google Drive API Test File")
        
    try:
        file_metadata = {
            "name": f"Test_Upload_{os.urandom(4).hex()}.txt",
            "parents": [folder_id]
        }
        media = MediaFileUpload(test_file_name, mimetype="text/plain")
        
        file = service.files().create(
            body=file_metadata, media_body=media, fields="id, webViewLink"
        ).execute()
        
        print(f"✅ 파일 업로드 성공!")
        print(f"   - File ID: {file.get('id')}")
        print(f"   - Link: {file.get('webViewLink')}")
        
    except Exception as e:
        print(f"❌ 파일 업로드 실패: {e}")
    finally:
        if os.path.exists(test_file_name):
            os.remove(test_file_name)

if __name__ == "__main__":
    test_upload()
