"""Google Drive 이미지 업로드 모듈"""

from __future__ import annotations

from pathlib import Path

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    Credentials = None
    build = None
    MediaFileUpload = None

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


def get_or_create_folder(service, folder_name: str) -> str:
    """
    Google Drive에서 폴더 찾기 또는 생성
    
    Args:
        service: Google Drive API 서비스 객체
        folder_name: 폴더명
        
    Returns:
        폴더 ID
    """
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    
    folders = results.get('files', [])
    if folders:
        print(f"[Drive] 기존 폴더 사용: {folder_name}")
        return folders[0]['id']
    
    # 폴더 생성
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    print(f"[Drive] 새 폴더 생성: {folder_name}")
    return folder['id']


def upload_to_drive(
    image_path: str,
    credentials_path: str,
    folder_name: str = "Stock_Chart_Screenshots",
    share_email: str | None = None,
) -> str | None:
    """
    이미지를 Google Drive에 업로드하고 공유 가능한 URL 반환
    
    Args:
        image_path: 로컬 이미지 파일 경로
        credentials_path: Google 서비스 계정 JSON 경로
        folder_name: Drive 폴더명
        share_email: 파일을 공유할 사용자 이메일 (선택)
        
    Returns:
        Google Drive 이미지 URL 또는 None
    """
    if build is None or Credentials is None or MediaFileUpload is None:
        print("[Drive] Google API 라이브러리가 설치되지 않았습니다")
        return None
    
    try:
        import os
        import json
        
        path = Path(image_path)
        if not path.exists():
            print(f"[Drive] 파일을 찾을 수 없음: {image_path}")
            return None
        
        # 서비스 계정 인증 (환경 변수 우선, 파일 fallback)
        service_account_json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        
        if service_account_json_str:
            # 환경 변수에서 로드 (GitHub Actions)
            service_account_info = json.loads(service_account_json_str)
            credentials = Credentials.from_service_account_info(
                service_account_info, scopes=SCOPES
            )
        elif credentials_path:
            # 파일에서 로드 (로컬)
            credentials = Credentials.from_service_account_file(
                credentials_path, scopes=SCOPES
            )
        else:
            print("[Drive] 서비스 계정 인증 정보가 없습니다")
            return None
        
        service = build('drive', 'v3', credentials=credentials)
        
        # 폴더 가져오기/생성
        folder_id = get_or_create_folder(service, folder_name)
        
        # 폴더를 사용자와 공유 (첫 실행 시)
        if share_email:
            try:
                service.permissions().create(
                    fileId=folder_id,
                    body={'type': 'user', 'role': 'writer', 'emailAddress': share_email},
                    sendNotificationEmail=False
                ).execute()
            except Exception as e:
                # 이미 공유되어 있으면 무시
                pass
        
        # 기존 파일 삭제 (중복 방지)
        file_name = path.name
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        existing = service.files().list(q=query, fields="files(id)").execute()
        
        for file in existing.get('files', []):
            service.files().delete(fileId=file['id']).execute()
            print(f"[Drive] 기존 파일 삭제: {file_name}")
        
        # 파일 업로드
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        media = MediaFileUpload(str(path), mimetype='image/png')
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        file_id = file['id']
        
        # 공개 공유 설정 (누구나 볼 수 있게)
        service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()
        
        # 사용자와 파일 공유 (선택)
        if share_email:
            try:
                service.permissions().create(
                    fileId=file_id,
                    body={'type': 'user', 'role': 'writer', 'emailAddress': share_email},
                    sendNotificationEmail=False
                ).execute()
            except Exception:
                pass
        
        # 직접 링크 생성 (=IMAGE()에서 사용 가능)
        image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
        
        print(f"[Drive] 업로드 성공: {file_name}")
        return image_url
        
    except Exception as e:
        import traceback
        print(f"[Drive] 업로드 에러 ({image_path}): {e}")
        print(f"[Drive] 상세 에러:\n{traceback.format_exc()}")
        return None
