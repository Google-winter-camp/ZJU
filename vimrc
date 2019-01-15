set nocompatible                            "vim��vi֧�ָ��๦�ܣ���showcmd�������ͻ�Ͳ����ݣ��رռ���
set nu                                        "��ʾ�к�
"set cindent                                    "c���Է����Զ�����
"set smartindent                                "�Ľ����cindent���Զ�ʶ����#��ͷ��ע�ͣ������л���
set autoindent                                "�������һ��������ݲ�ͬ�������ͽ��в�ͬ������������
set tabstop=4                                "����tab����Ϊ4
set shiftwidth=4                            "�����Զ��������������
set background=dark                            "solarized����ɫģʽ
set mouse=a                                    "������buffer���κεط�ʹ�����
set encoding=utf-8
set history=100                                "ָ���¼��Ĭ��20��
set showcmd                                    "��ʾ��������

syntax enable
syntax on                                    "�﷨����

"F5����
augroup ccompile
    autocmd Filetype c map <F5> <Esc>:w<CR>:!gcc % -Wall -std=c11 -o %< -lm <CR>
    autocmd Filetype cpp map <F5> <Esc>:w<CR>:!g++ % -Wall -std=c++14 -o %< <CR>
    autocmd Filetype python map <F5> <Esc>:w<CR>:!python % <CR>
augroup END

"F6����
augroup crun
    autocmd Filetype c map <F6> <Esc>:!time ./%< <CR>
    autocmd Filetype cpp map <F6> <Esc>:!time ./%< <CR>
augroup END

"���������Զ�ƥ��
inoremap ( ()<Esc>i
inoremap [ []<Esc>i
inoremap { {}<Esc>i
inoremap ' ''<Esc>i
inoremap " ""<Esc>i

"Ctrl+Aȫѡ Ctrl+C���� Ctrl+Vճ��
map <C-A> ggvG$
imap <C-A> <Esc>ggvG$
vmap <C-C> "+y<Esc>
map <C-V> "+p
imap <C-V> <Esc>"+pa