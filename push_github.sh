#!/usr/bin/env bash
# Chạy bằng Bash trên Linux, macOS hoặc Git Bash (Windows).
# Cách dùng, tại thư mục gốc source code:
#   bash push_github.sh https://github.com/USERNAME/REPO.git
#   bash push_github.sh git@github.com:USERNAME/REPO.git "Update source code"
# Repo phải được tạo sẵn trên GitHub. Lần đầu nên dùng repo trống.
# Nếu không truyền URL, script dùng origin hiện tại hoặc hỏi URL.
# Giữ nhánh hiện tại; source chưa có Git sẽ dùng nhánh main.
# Script commit tất cả thay đổi không bị ignore, bao gồm file bị xóa.
# Không tự merge/rebase, không force push, không nhúng token vào URL.

set -euo pipefail
die() { printf '\nLỖI: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nThao tác Git thất bại. Kiểm tra thông báo phía trên rồi chạy lại.\n" >&2' ERR

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,12p' "$0"
  exit 0
fi
[[ $# -le 2 ]] || die "Cách dùng: bash push_github.sh [URL_REPO] [COMMIT_MESSAGE]"
command -v git >/dev/null 2>&1 || die "Máy chưa cài Git."

repo_url="${1:-}"
commit_message="${2:-Update source code}"
[[ -n "$commit_message" ]] || die "Nội dung commit không được trống."
project_dir="$(pwd -P)"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_root="$(cd "$(git rev-parse --show-toplevel)" && pwd -P)"
  [[ "$git_root" == "$project_dir" ]] || die "Hãy chạy tại thư mục gốc repo: $git_root"
else
  [[ ! -e .git ]] || die "Có .git nhưng repo không hợp lệ."
fi

if [[ -z "$repo_url" ]]; then
  repo_url="$(git remote get-url origin 2>/dev/null || true)"
fi
if [[ -z "$repo_url" ]]; then
  read -r -p "URL repo GitHub: " repo_url || die "Cần cung cấp URL repo."
fi
# Chỉ chấp nhận GitHub HTTPS hoặc SSH, không cho token trong URL.
https_pattern='^https://github[.]com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$'
ssh_pattern='^git@github[.]com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$'
[[ "$repo_url" =~ $https_pattern || "$repo_url" =~ $ssh_pattern ]] ||
  die "URL hợp lệ: https://github.com/USERNAME/REPO.git hoặc git@github.com:USERNAME/REPO.git"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init
  git symbolic-ref HEAD refs/heads/main
fi
branch="$(git symbolic-ref --quiet --short HEAD)" || die "Đang detached HEAD. Hãy chuyển sang một nhánh trước."
for state in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply; do
  [[ ! -e "$(git rev-parse --git-path "$state")" ]] || die "Có merge/rebase/cherry-pick chưa hoàn tất."
done
[[ -z "$(git ls-files -u)" ]] || die "Có file đang xung đột. Hãy xử lý trước."

if git remote get-url origin >/dev/null 2>&1; then
  [[ "$(git remote get-url origin)" == "$repo_url" ]] ||
    die "origin đang trỏ tới URL khác. Script giữ nguyên; kiểm tra bằng git remote -v."
  [[ "$(git remote get-url --push origin)" == "$repo_url" ]] ||
    die "URL dùng để push khác URL repo. Kiểm tra bằng git remote -v."
  [[ "$(git remote get-url --push --all origin | wc -l | tr -d ' ')" == 1 ]] ||
    die "origin có nhiều URL push. Hãy cấu hình lại trước."
else
  git remote add origin "$repo_url"
fi

printf '\nKiểm tra kết nối và nhánh %s trên GitHub...\n' "$branch"
remote_ref="$(git ls-remote --heads origin "refs/heads/$branch")" ||
  die "Không truy cập được repo. Kiểm tra URL, quyền truy cập và đăng nhập Git/SSH trên máy."
if [[ -n "$remote_ref" ]]; then
  git fetch --no-tags origin "refs/heads/$branch"
  git rev-parse --verify HEAD >/dev/null 2>&1 ||
    die "Nhánh trên GitHub đã có commit. Hãy clone repo rồi chép source vào để giữ lịch sử."
  git merge-base --is-ancestor FETCH_HEAD HEAD ||
    die "GitHub có commit chưa có ở máy. Hãy đồng bộ nhánh, giải quyết xung đột rồi chạy lại."
fi

# Thêm các quy tắc ignore thông dụng; giữ nguyên nội dung .gitignore sẵn có.
ignore_marker='# BEGIN push_github.sh defaults'
if ! grep -Fqx "$ignore_marker" .gitignore 2>/dev/null; then
  cat >> .gitignore <<'IGNORE'

# BEGIN push_github.sh defaults
.env
.env.*
!.env.example
!.env.sample
!.env.template
.venv/
venv/
node_modules/
__pycache__/
*.pyc
.DS_Store
.next/
# Dự án ML: có thể bỏ dấu # ở các dòng sau nếu chỉ đẩy source code.
# datasets/
# checkpoints/
# *.pt
# *.pth
# END push_github.sh defaults
IGNORE
fi

# Kiểm tra cả file đã tracked: .gitignore không loại các file này.
# Đây là kiểm tra tên file, không phải công cụ dò mọi loại secret.
blocked=0
while IFS= read -r -d '' file; do
  base="${file##*/}"
  case "$base" in
    .env.example|.env.sample|.env.template) ;;
    .env|.env.*|id_rsa|id_ed25519|*.pem|*.key|credentials.json|service-account*.json)
      printf 'File có thể chứa thông tin nhạy cảm: %q\n' "$file" >&2
      blocked=1 ;;
  esac
done < <(git ls-files -z --cached --others --exclude-standard)
[[ "$blocked" == 0 ]] || die "Loại file nhạy cảm khỏi Git trước. Nếu đã commit secret, cần xử lý cả lịch sử và thu hồi secret."

printf '\nThư mục: %s\nRepo: %s\nNhánh: %s\nCommit: %s\n\n' "$project_dir" "$repo_url" "$branch" "$commit_message"
git status --short
printf '\nScript sẽ stage TẤT CẢ thay đổi không bị ignore, tạo commit và push.\n'
read -r -p "Tiếp tục? [y/N]: " answer || die "Không nhận được xác nhận."
case "$answer" in y|Y|yes|YES) ;; *) printf 'Đã dừng; chưa stage, commit hoặc push.\n'; exit 0 ;; esac

for field in name email; do
  if [[ -z "$(git config --get "user.$field" || true)" ]]; then
    read -r -p "Git user.$field (chỉ lưu cho repo này): " identity || die "Thiếu Git user.$field."
    [[ -n "$identity" ]] || die "Git user.$field không được trống."
    git config --local "user.$field" "$identity"
  fi
done

git add --all -- .
# Chính sách của script: chặn blob trên 95 MiB trong lịch sử của nhánh
# sẽ được push, kể cả file lớn đã xóa khỏi phiên bản hiện tại.
object_ids="$(mktemp)"
trap 'rm -f "$object_ids"' EXIT
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  git rev-list --objects HEAD | awk '{print $1}' >> "$object_ids"
fi
git ls-files --stage | awk '{print $2}' >> "$object_ids"
large_objects="$(sort -u "$object_ids" | git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize)' | awk '$2 == "blob" && $3 > 99614720 {print $1 " (" $3 " bytes)"}')"
[[ -z "$large_objects" ]] || die "Có blob trên 95 MiB trong index/lịch sử nhánh:
$large_objects
Hãy dùng Git LFS hoặc loại file lớn khỏi lịch sử. Các thay đổi đã stage được giữ để bạn xử lý."

if git diff --cached --quiet; then
  printf '\nKhông có thay đổi mới để commit.\n'
else
  git commit -m "$commit_message"
fi
git rev-parse --verify HEAD >/dev/null 2>&1 || die "Repo chưa có commit để push."
git push --no-follow-tags -u origin "HEAD:refs/heads/$branch" ||
  die "Push thất bại. Commit ở máy vẫn được giữ; kiểm tra đăng nhập, quyền repo, hoặc thay đổi mới trên GitHub."
printf '\nĐã đẩy source code lên GitHub thành công!\n'
