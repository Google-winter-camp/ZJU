set nocompatible                            "vim比vi支持更多功能，如showcmd，避免冲突和不兼容，关闭兼容
set nu                                        "显示行号
"set cindent                                    "c语言风格的自动缩进
"set smartindent                                "改进版的cindent，自动识别以#开头的注释，不进行换行
set autoindent                                "配合下面一条命令根据不同语言类型进行不同缩进，更智能
set tabstop=4                                "设置tab长度为4
set shiftwidth=4                            "设置自动对齐的缩进级别
set background=dark                            "solarized的深色模式
set mouse=a                                    "可以找buffer的任何地方使用鼠标
set encoding=utf-8
set history=100                                "指令记录，默认20行
set showcmd                                    "显示输入命令

syntax enable
syntax on                                    "语法高亮

"F5编译
augroup ccompile
    autocmd Filetype c map <F5> <Esc>:w<CR>:!gcc % -Wall -std=c11 -o %< -lm <CR>
    autocmd Filetype cpp map <F5> <Esc>:w<CR>:!g++ % -Wall -std=c++14 -o %< <CR>
    autocmd Filetype python map <F5> <Esc>:w<CR>:!python % <CR>
augroup END

"F6运行
augroup crun
    autocmd Filetype c map <F6> <Esc>:!time ./%< <CR>
    autocmd Filetype cpp map <F6> <Esc>:!time ./%< <CR>
augroup END

"括号引号自动匹配
inoremap ( ()<Esc>i
inoremap [ []<Esc>i
inoremap { {}<Esc>i
inoremap ' ''<Esc>i
inoremap " ""<Esc>i

"Ctrl+A全选 Ctrl+C复制 Ctrl+V粘贴
map <C-A> ggvG$
imap <C-A> <Esc>ggvG$
vmap <C-C> "+y<Esc>
map <C-V> "+p
imap <C-V> <Esc>"+pa